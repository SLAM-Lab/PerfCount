#!/usr/bin/env python3
"""
Processes warmup perf .out files into warmup_params.csv.

Input files produced by warmup_collection.sh (one file per direction):
    cpu_16_{freq}GHz_dacapo_{bench}_10000000_warmup_PtoE.out  (E-core, destination)
    cpu_0_{freq}GHz_dacapo_{bench}_10000000_warmup_EtoP.out   (P-core, destination)

Each file was collected with -a -C {dest_cpu}. Before migration it contains only
near-zero instruction counts (destination core idle). After migration the benchmark
runs there, producing full 10M-instruction chunks.

For each (benchmark, direction) pair:
  1. Parses the destination-core .out file
  2. Identifies migration chunk M: first chunk with instructions >= 5M
  3. Extracts post-migration chunks [M : M+N_POST_CHUNKS]
  4. Loads steady-state ref_cycles from the existing aligned CSV for the destination core
  5. Computes slowdown[k] = post_ref_cycles[k] / mean_steady_ref_cycles
  6. Fits slowdown(k) = 1 + A * exp(-k / tau) via scipy.optimize.curve_fit
  7. Collects LLC miss rates (warmup window vs steady state)

Output: warmup_params.csv
    benchmark, direction, A, tau, llc_miss_warmup, llc_miss_steady, fit_rmse, n_post_chunks
"""

import os
import re
import sys
import argparse
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import curve_fit

# Import reusable parsing functions from process_raw_data.py (same directory)
sys.path.insert(0, str(Path(__file__).parent))
from process_raw_data import parse_perf_script_output

# ── Constants ─────────────────────────────────────────────────────────────────
P_CPU = 0
E_CPU = 16
MIGRATION_INSTR_THRESHOLD = 5_000_000   # 50% of 10M; post-migration chunk when exceeded
N_POST_CHUNKS = 100                     # post-migration chunks to extract for curve fitting

ALIGNED_CSV_DIR_DEFAULT = "../../../processed_data_100M/x86_desktop_heterogeneous"

WARMUP_FILE_RE = re.compile(
    r"cpu_(?P<cpu_id>\d+)_(?P<freq>[\d\.]+)GHz_"
    r"(?P<suite>dacapo|spec)_(?P<bench>.+)_10000000_warmup_"
    r"(?P<direction>PtoE|EtoP)\.out"
)


def load_parsed_df(out_file: str) -> pd.DataFrame:
    cmd = ["perf", "script", "-i", out_file]
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
        df = parse_perf_script_output(proc.stdout, arch="x86")
    return df


def find_migration_chunk(df: pd.DataFrame) -> int:
    """Return index of first row where instructions >= MIGRATION_INSTR_THRESHOLD."""
    if 'instructions' not in df.columns:
        return 0
    mask = df['instructions'] >= MIGRATION_INSTR_THRESHOLD
    if not mask.any():
        return 0
    return int(mask.idxmax())


def steady_state_ref_cycles(bench: str, suite: str, freq: str,
                             dest_cpu: int, aligned_csv_dir: str) -> float:
    """Mean ref_cycles/chunk (skip chunk 0) from aligned CSV for destination core."""
    fname = f"aligned_{suite}_{bench}_{freq}GHz_cpu{dest_cpu}_phase0.csv"
    path  = Path(aligned_csv_dir) / fname
    if not path.exists():
        raise FileNotFoundError(f"Steady-state CSV not found: {path}")
    df = pd.read_csv(path)
    if 'ref_cycles' not in df.columns:
        raise KeyError(f"'ref_cycles' missing in {path}")
    return float(df['ref_cycles'].iloc[1:].mean())


def fit_warmup_curve(slowdown: np.ndarray):
    """Fit 1 + A * exp(-k / tau) to slowdown array. Returns (A, tau, rmse)."""
    ks = np.arange(len(slowdown), dtype=float)

    def model(k, A, tau):
        return 1.0 + A * np.exp(-k / tau)

    try:
        popt, _ = curve_fit(
            model, ks, slowdown,
            p0=[0.3, 5.0],
            bounds=([0.0, 0.5], [3.0, 100.0]),
            maxfev=10000
        )
        A, tau = float(popt[0]), float(popt[1])
        rmse = float(np.sqrt(np.mean((slowdown - model(ks, A, tau)) ** 2)))
        return A, tau, rmse
    except RuntimeError:
        return 0.0, 1.0, float('nan')


def llc_miss_rate(df: pd.DataFrame) -> float:
    """Mean LLC miss rate = llc_misses / llc_loads."""
    if 'llc_loads' not in df.columns or 'llc_misses' not in df.columns:
        return float('nan')
    loads = df['llc_loads'].mean()
    return float(df['llc_misses'].mean() / loads) if loads > 0 else float('nan')


def process_pair(bench: str, suite: str, freq: str, direction: str,
                 dest_cpu: int, raw_dir: str, aligned_csv_dir: str) -> dict | None:
    """Process one (benchmark, direction) pair and return a result dict or None."""
    out_file = os.path.join(
        raw_dir, f"cpu_{dest_cpu}_{freq}GHz_{suite}_{bench}_10000000_warmup_{direction}.out"
    )
    if not os.path.exists(out_file):
        print(f"  [SKIP] Missing: {out_file}")
        return None

    print(f"  {suite}_{bench} {direction} ...", flush=True)
    df = load_parsed_df(out_file)

    if df.empty or 'ref_cycles' not in df.columns:
        print(f"  [WARN] No usable data in {out_file}")
        return None

    mig_idx = find_migration_chunk(df)
    post_df = df.iloc[mig_idx : mig_idx + N_POST_CHUNKS].reset_index(drop=True)

    if len(post_df) < 10:
        print(f"  [WARN] Only {len(post_df)} post-migration chunks in {out_file} — skipping")
        return None

    try:
        steady_rc = steady_state_ref_cycles(bench, suite, freq, dest_cpu, aligned_csv_dir)
    except (FileNotFoundError, KeyError) as ex:
        print(f"  [WARN] {ex}")
        return None

    slowdown = post_df['ref_cycles'].values / steady_rc
    A, tau, rmse = fit_warmup_curve(slowdown)

    warmup_llc = llc_miss_rate(post_df)
    try:
        steady_path = (Path(aligned_csv_dir) /
                       f"aligned_{suite}_{bench}_{freq}GHz_cpu{dest_cpu}_phase0.csv")
        steady_df  = pd.read_csv(steady_path)
        steady_llc = llc_miss_rate(steady_df.iloc[1:])
    except Exception:
        steady_llc = float('nan')

    return {
        'benchmark':       f"{suite}_{bench}",
        'direction':       direction,
        'A':               round(A,           4),
        'tau':             round(tau,          3),
        'llc_miss_warmup': round(warmup_llc,   4),
        'llc_miss_steady': round(steady_llc,   4),
        'fit_rmse':        round(rmse,         4),
        'n_post_chunks':   len(post_df),
    }


def dest_cpu_for(direction: str) -> int:
    return E_CPU if direction == 'PtoE' else P_CPU


def discover_pairs(raw_dir: str) -> list:
    """Scan raw_dir for warmup .out files and return sorted list of tuples."""
    pairs = {}
    for fname in os.listdir(raw_dir):
        m = WARMUP_FILE_RE.match(fname)
        if m:
            key = (m.group('bench'), m.group('suite'),
                   m.group('freq'), m.group('direction'))
            pairs[key] = True
    return sorted(pairs.keys())


def main():
    parser = argparse.ArgumentParser(
        description="Process warmup perf .out files into exponential-fit parameters."
    )
    parser.add_argument("--raw_dir",         default=".",
                        help="Directory containing warmup .out files")
    parser.add_argument("--aligned_csv_dir", default=ALIGNED_CSV_DIR_DEFAULT,
                        help="Directory with steady-state aligned CSVs")
    parser.add_argument("--out",             default="warmup_params.csv",
                        help="Output CSV path")
    args = parser.parse_args()

    pairs = discover_pairs(args.raw_dir)
    if not pairs:
        print(f"No warmup .out files found in {args.raw_dir}")
        return

    print(f"Found {len(pairs)} (benchmark, direction) pairs.")
    results = []
    for bench, suite, freq, direction in pairs:
        dest_cpu = dest_cpu_for(direction)
        row = process_pair(bench, suite, freq, direction,
                           dest_cpu, args.raw_dir, args.aligned_csv_dir)
        if row:
            results.append(row)

    if results:
        df = pd.DataFrame(results)
        df.to_csv(args.out, index=False)
        print(f"\nWrote {len(results)} rows to {args.out}")
        print(df.to_string(index=False))
    else:
        print("No valid results produced.")


if __name__ == "__main__":
    main()

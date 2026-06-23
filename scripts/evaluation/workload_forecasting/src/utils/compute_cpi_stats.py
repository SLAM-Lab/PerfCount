#!/usr/bin/env python3
"""
compute_cpi_stats.py
====================
Compute CPI (Cycles Per Instruction) variability statistics from processed
data files. CPI variability is a measure of how hard it is to forecast
upcoming counter values: a workload with highly variable CPI is inherently
less predictable.

Statistics are computed per (workload_base, freq) by aggregating across all
phase files that belong to that workload/frequency combination.

Output columns
--------------
  machine         : machine identifier (e.g. arm_server)
  workload_base   : benchmark name without phase (e.g. spec_500.perlbench_r)
  freq            : source frequency in GHz
  cpu             : cpu id if present in filename, else ''
  n_phases        : number of phase files found
  n_samples       : total 100M-instruction samples across all phases
  cpi_mean        : mean CPI across all samples
  cpi_std         : standard deviation of CPI
  cpi_min         : minimum CPI observed
  cpi_max         : maximum CPI observed
  cpi_range       : cpi_max - cpi_min
  cpi_cv          : coefficient of variation = cpi_std / cpi_mean
                    (scale-free difficulty metric; higher = harder to predict)
  cpi_p10         : 10th percentile CPI
  cpi_p90         : 90th percentile CPI
  cpi_iqr         : interquartile range (p75 - p25)

Usage
-----
  # Single machine directory
  python compute_cpi_stats.py \\
      --data_dir ~/PerfCount/processed_data_10M/arm_server \\
      --out cpi_stats_arm_server_10M.csv

  # Multiple directories in one pass
  python compute_cpi_stats.py \\
      --data_dir ~/PerfCount/processed_data_100M/arm_edge_heterogeneous \\
                 ~/PerfCount/processed_data_100M/x86_desktop_heterogeneous \\
      --out cpi_stats_100M.csv

Joining with forecasting results
---------------------------------
  import pandas as pd, re

  forecast = pd.read_csv("condensed_results_100M.csv")
  stats    = pd.read_csv("cpi_stats_100M.csv")

  # Extract workload_base and freq from the forecasting workload column:
  #   e.g. "aligned_spec_500.perlbench_r_1.0GHz_phase0"
  #        -> workload_base = "spec_500.perlbench_r", freq = "1.0"
  def parse_wl(name):
      name = re.sub(r'^aligned_', '', name)
      m = re.match(r'(.+?)_([\d.]+)GHz', name)
      return (m.group(1), m.group(2)) if m else (name, '')

  forecast[['workload_base','freq']] = forecast['workload'].apply(
      lambda x: pd.Series(parse_wl(x)))
  merged = forecast.merge(
      stats[['workload_base','freq','cpi_cv','cpi_range','cpi_mean']],
      on=['workload_base','freq'], how='left')
"""

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd


FILENAME_PATTERN = re.compile(
    r"aligned_(?P<bench>.+?)_(?P<freq>[\d.]+)GHz"
    r"(?:_cpu(?P<cpu>\d+))?_phase(?P<phase>\d+)\.csv"
)

# Human-readable labels for known (machine, cpu_id) pairs.
# Falls back to "cpu{id}" for anything not listed here.
CPU_LABELS = {
    ("x86_desktop_heterogeneous", "0"):  "pcore",
    ("x86_desktop_heterogeneous", "16"): "ecore",
    ("arm_edge_heterogeneous",    "1"):  "ino",
    ("arm_edge_heterogeneous",    "4"):  "ooo",
}


def _machine_name(data_dir):
    return os.path.basename(os.path.normpath(data_dir))


def compute_stats_for_dir(data_dir):
    """
    Scan data_dir for aligned CSV files, compute per-(workload_base, freq, cpu)
    CPI statistics.

    Returns a DataFrame with one row per (workload_base, freq, cpu).
    """
    machine = _machine_name(data_dir)
    files = sorted(glob.glob(os.path.join(data_dir, "**", "aligned_*.csv"), recursive=True))
    if not files:
        print(f"  [WARN] No aligned_*.csv files in {data_dir}")
        return pd.DataFrame()

    # Group files by (bench, freq, cpu)
    groups = {}
    for fpath in files:
        m = FILENAME_PATTERN.match(os.path.basename(fpath))
        if not m:
            continue
        key = (m.group("bench"), m.group("freq"), m.group("cpu") or "")
        groups.setdefault(key, []).append(fpath)

    rows = []
    for (bench, freq, cpu), paths in sorted(groups.items()):
        cpi_vals = []
        for fpath in paths:
            try:
                df = pd.read_csv(fpath)
                df.columns = [c.strip() for c in df.columns]
                if "cpu_cycles" not in df.columns or "instructions" not in df.columns:
                    continue
                valid = df[(df["instructions"] > 0) & (df["cpu_cycles"] > 0)].copy()
                if valid.empty:
                    continue
                cpi_vals.append(valid["cpu_cycles"].values / valid["instructions"].values)
            except Exception as e:
                print(f"  [WARN] Could not read {os.path.basename(fpath)}: {e}")

        if not cpi_vals:
            continue

        all_cpi = np.concatenate(cpi_vals)
        cpu_label = CPU_LABELS.get((machine, cpu), f"cpu{cpu}" if cpu else "")
        rows.append({
            "machine":      machine,
            "workload_base": bench,
            "freq":         freq,
            "cpu":          cpu,
            "cpu_label":    cpu_label,
            "n_phases":     len(cpi_vals),
            "n_samples":    len(all_cpi),
            "cpi_mean":     round(float(np.mean(all_cpi)),    4),
            "cpi_std":      round(float(np.std(all_cpi)),     4),
            "cpi_min":      round(float(np.min(all_cpi)),     4),
            "cpi_max":      round(float(np.max(all_cpi)),     4),
            "cpi_range":    round(float(np.max(all_cpi) - np.min(all_cpi)), 4),
            "cpi_cv":       round(float(np.std(all_cpi) / np.mean(all_cpi)) if np.mean(all_cpi) > 0 else 0, 4),
            "cpi_p10":      round(float(np.percentile(all_cpi, 10)), 4),
            "cpi_p90":      round(float(np.percentile(all_cpi, 90)), 4),
            "cpi_iqr":      round(float(np.percentile(all_cpi, 75) - np.percentile(all_cpi, 25)), 4),
        })

    df_out = pd.DataFrame(rows)
    print(f"  {machine}: {len(df_out)} (workload_base, freq) combinations "
          f"from {len(files)} files")
    return df_out


def main():
    parser = argparse.ArgumentParser(description="Compute CPI variability stats from processed data.")
    parser.add_argument("--data_dir", nargs="+", required=True,
                        help="One or more processed_data_* directories to scan.")
    parser.add_argument("--out", required=True,
                        help="Output CSV path.")
    args = parser.parse_args()

    all_dfs = []
    for d in args.data_dir:
        if not os.path.isdir(d):
            print(f"[WARN] Not a directory, skipping: {d}")
            continue
        print(f"Processing {d} ...")
        df = compute_stats_for_dir(d)
        if not df.empty:
            all_dfs.append(df)

    if not all_dfs:
        print("[ERROR] No data collected.")
        return

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(args.out, index=False)
    print(f"\nWrote {len(combined)} rows → {args.out}")

    # Quick summary
    print(f"\nTop 10 most variable workloads (by cpi_cv):")
    print(combined.nlargest(10, "cpi_cv")[
        ["machine", "cpu_label", "workload_base", "freq", "cpi_mean", "cpi_cv", "cpi_range", "n_phases"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()

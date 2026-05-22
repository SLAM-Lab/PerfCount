#!/usr/bin/env python3
"""
compute_cv_instructions.py
==========================
Measures coefficient of variation (CV = std/mean * 100%) in total
instruction count across repeated runs of each DaCapo benchmark.

"Total instruction count" = number of 10M-instruction samples in a run,
i.e. the trace length.  Per-sample instruction count is ~10M by construction
(perf fires every 10M instructions), so CV of per-sample count is
near-zero and uninformative.  CV of trace length captures how consistently
the benchmark executes the same total amount of work across runs — which
separates deterministic single-threaded benchmarks from server-oriented
workloads whose execution time (and JIT compilation load) varies.

For each (benchmark, frequency):
  - Counts the number of instruction-sampling intervals in each run
    (C2 JIT-compiler samples excluded, consistent with process_raw_data_no_c2).
  - Computes CV = std / mean * 100% across runs.

Final output: one row per benchmark with mean CV averaged across
frequencies, sorted from lowest to highest CV.

Run from ~/PerfCount/:
    python scripts/data_processing/arm_server/compute_cv_instructions.py
"""

import os
import re
import glob
import subprocess
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

RAW_DIR  = "raw_data/arm_server"
FILE_PAT = re.compile(
    r"cpu_(?P<cpu_id>\d+)_(?P<freq>[\d.]+)GHz_(?P<bench>dacapo_.+)_10000000_(?P<run>\d+)_(?P<phase>\d+)\.out"
)

# ---------------------------------------------------------------------------
# Count instruction-sampling intervals in one raw .out file
# ---------------------------------------------------------------------------

def count_samples(path):
    """Return the number of instruction-sampling intervals in a perf .out file."""
    cmd = ["perf", "script", "-i", path]
    seen_ts = set()

    try:
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) as proc:
            for raw_line in proc.stdout:
                line = raw_line.decode("utf-8", errors="replace")
                if not line.strip() or line.lstrip().startswith("#"):
                    continue

                parts = line.split()
                if len(parts) < 4:
                    continue

                # Skip C2 JIT compiler thread samples
                if parts[0] == "C2":
                    continue

                # Only count lines that record the instructions event
                for i, p in enumerate(parts):
                    if p.endswith(":"):
                        try:
                            float(p[:-1])
                        except ValueError:
                            continue
                        if i + 2 >= len(parts):
                            break
                        evt = parts[i + 2].lower().split(":")[0]
                        if evt in ("instructions", "instructions_retired"):
                            seen_ts.add(p)
                        break

    except Exception:
        pass

    return len(seen_ts)


def _worker(args):
    path, meta = args
    n = count_samples(path)
    return meta["bench"], meta["freq"], int(meta["run"]), n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    files = sorted(glob.glob(os.path.join(RAW_DIR, "*dacapo*.out")))
    if not files:
        raise SystemExit(f"No dacapo .out files found in {RAW_DIR}")

    print(f"Found {len(files)} raw dacapo files — counting instruction samples...")

    tasks = []
    for f in files:
        m = FILE_PAT.search(os.path.basename(f))
        if m:
            tasks.append((f, m.groupdict()))

    # Collect sample counts in parallel
    raw = {}  # (bench, freq) -> {run: sample_count}
    n_workers = min(40, len(tasks))
    with ProcessPoolExecutor(max_workers=n_workers) as exe:
        futs = {exe.submit(_worker, t): t for t in tasks}
        done = 0
        for fut in as_completed(futs):
            bench, freq, run, n = fut.result()
            raw.setdefault((bench, freq), {})[run] = n
            done += 1
            if done % 20 == 0:
                print(f"  {done}/{len(tasks)} files done...")

    print(f"Done.\n")

    # Compute CV per (bench, freq), then average across freqs
    bench_rows = {}  # bench -> list of (freq, mean_samples, cv)

    for (bench, freq), run_dict in sorted(raw.items()):
        counts = np.array([run_dict[r] for r in sorted(run_dict)], dtype=float)
        if len(counts) < 2 or counts.mean() == 0:
            continue
        cv = counts.std(ddof=1) / counts.mean() * 100.0
        bench_rows.setdefault(bench, []).append((freq, counts.mean(), cv))

    rows = []
    for bench, entries in bench_rows.items():
        mean_cv      = np.mean([e[2] for e in entries])
        mean_samples = np.mean([e[1] for e in entries])
        rows.append({
            "benchmark":    bench.replace("dacapo_", ""),
            "mean_cv_pct":  mean_cv,
            "mean_samples": mean_samples,
            "n_freqs":      len(entries),
        })

    df = pd.DataFrame(rows).sort_values("mean_cv_pct").reset_index(drop=True)

    print("=" * 60)
    print(f"{'Benchmark':<20} {'Mean CV (%)':>12}  {'Mean samples':>14}")
    print("-" * 60)
    for _, row in df.iterrows():
        print(f"  {row['benchmark']:<18} {row['mean_cv_pct']:>12.3f}  {row['mean_samples']:>14,.0f}")
    print("=" * 60)
    print(f"\nOverall mean CV : {df['mean_cv_pct'].mean():.3f}%")
    print(f"Min (most deterministic) : {df['mean_cv_pct'].min():.3f}%  ({df.iloc[0]['benchmark']})")
    print(f"Max (least deterministic): {df['mean_cv_pct'].max():.3f}%  ({df.iloc[-1]['benchmark']})")

    out_path = "results/dacapo_cv_instructions_arm_server.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()

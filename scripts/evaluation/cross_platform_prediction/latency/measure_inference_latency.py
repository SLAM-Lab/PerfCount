#!/usr/bin/env python3
"""
measure_inference_latency.py
============================
Single-sample inference latency for trained cross-platform CatBoost models
across all (exp_type, platform, suite, feature_set) combinations at a chosen
granularity (default: 10M).

Overhead removal:
  - thread_count=1  (no internal CatBoost parallelism)
  - Raw numpy float32 array input  (no DataFrame, no Python dict conversion)
  - CPU affinity pinned to one core  (eliminates scheduler-migration noise)
  - GC disabled during the hot loop
  - time.perf_counter_ns()  (nanosecond resolution, no system-call overhead)
  - Adaptive rep count: more reps for fast models, fewer for slow ones

One representative model is picked per (exp_type, platform, suite, feature_set)
— the alphabetically first trained workload model — so results reflect a real
tree depth / tree count from the actual training run.

Outputs
-------
  results/cross_platform/latency/inference_latency_{gran}.csv
  results/cross_platform/latency/pareto_{gran}.csv   (if grand_summary.csv found)

Usage
-----
  python measure_inference_latency.py                   # default: 10M
  python measure_inference_latency.py --gran 100M       # coarser windows
  python measure_inference_latency.py --reps 200000     # more reps
  python measure_inference_latency.py --cpu 3           # pin to cpu 3
"""

import argparse
import gc
import glob
import os
import time

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FEATURE_SETS = ["full", "top6", "top4"]
EXP_TYPES    = ["cross_freq", "cross_proc"]

# Adaptive rep targets: aim for this many µs of total timing per model.
# Results in ~100k reps for 1µs models, ~10k for 10µs models.
TARGET_TIMING_US = 200_000
WARMUP_REPS      = 500
MIN_REPS         = 5_000
MAX_REPS         = 500_000


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_models(results_root, gran, feature_sets):
    """
    Find one representative model per (exp_type, platform, suite, feature_set).
    Walks every *_{gran} subdirectory under each experiment type and identifies
    feature_set by matching a path component. Returns a list of dicts sorted by
    (exp_type, platform, suite, feature_set).
    """
    groups = {}
    for exp_type in EXP_TYPES:
        exp_dir = os.path.join(results_root, exp_type)
        if not os.path.isdir(exp_dir):
            continue
        for cbm in sorted(glob.glob(os.path.join(exp_dir, "**", "model_*.cbm"), recursive=True)):
            rel   = os.path.relpath(cbm, exp_dir)
            parts = rel.split(os.sep)
            # parts[0] = platform (must end with _{gran})
            if not parts[0].endswith(f"_{gran}"):
                continue
            platform = parts[0]
            # Find feature_set token in the remaining path parts
            for i, p in enumerate(parts[1:], start=1):
                if p in feature_sets:
                    # the path component immediately before feature_set is the suite
                    suite = parts[i - 1] if i >= 2 else "unknown"
                    key   = (exp_type, platform, suite, p)
                    groups.setdefault(key, cbm)   # keep first (alphabetically)
                    break

    missing = set(feature_sets) - {k[3] for k in groups}
    if missing:
        print(f"  [INFO] No models found for feature sets: {sorted(missing)}")

    return [
        {"exp_type": k[0], "platform": k[1], "suite": k[2],
         "feature_set": k[3], "model_path": v}
        for k, v in sorted(groups.items())
    ]


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def pin_cpu(cpu_id):
    """Pin the calling process to a single CPU core (Linux only)."""
    try:
        os.sched_setaffinity(0, {cpu_id})
        print(f"  Pinned to CPU {cpu_id}")
    except AttributeError:
        print("  [WARN] os.sched_setaffinity not available on this OS — skipping CPU pin")
    except OSError as e:
        print(f"  [WARN] CPU pin failed: {e}")


def build_input(model):
    """
    Single-row numpy float32 array matching the model's feature schema.
    All-float, no categoricals — so numpy is legal and eliminates all
    DataFrame construction overhead. Value 1.0 is representative of
    normalized counter rates and traverses real tree branches.
    """
    n = len(model.feature_names_)
    return np.ones((1, n), dtype=np.float32)


def time_model(model, X, reps, warmup):
    """
    Time `reps` single-sample predict() calls.
    GC is disabled for the duration of the hot loop.
    Returns latency array in µs.
    """
    # Warmup: let CatBoost JIT any internal caches, warm the instruction cache
    for _ in range(warmup):
        model.predict(X, thread_count=1)

    gc.disable()
    try:
        times_ns = np.empty(reps, dtype=np.int64)
        for i in range(reps):
            t0 = time.perf_counter_ns()
            model.predict(X, thread_count=1)
            times_ns[i] = time.perf_counter_ns() - t0
    finally:
        gc.enable()

    return times_ns / 1e3   # → µs


def adaptive_reps(model, X, warmup):
    """
    Run a small pilot (1000 calls) to estimate latency, then compute
    how many reps to run to reach TARGET_TIMING_US of total timing.
    """
    for _ in range(warmup // 5):
        model.predict(X, thread_count=1)
    t0 = time.perf_counter_ns()
    for _ in range(1000):
        model.predict(X, thread_count=1)
    elapsed_us = (time.perf_counter_ns() - t0) / 1e3
    mean_us    = elapsed_us / 1000
    reps = int(TARGET_TIMING_US / max(mean_us, 0.01))
    return max(MIN_REPS, min(MAX_REPS, reps))


def summarize(times_us):
    return {
        "latency_mean_us":   float(np.mean(times_us)),
        "latency_std_us":    float(np.std(times_us)),
        "latency_median_us": float(np.median(times_us)),
        "latency_p50_us":    float(np.percentile(times_us, 50)),
        "latency_p90_us":    float(np.percentile(times_us, 90)),
        "latency_p95_us":    float(np.percentile(times_us, 95)),
        "latency_p99_us":    float(np.percentile(times_us, 99)),
        "latency_p999_us":   float(np.percentile(times_us, 99.9)),
        "latency_min_us":    float(np.min(times_us)),
        "latency_max_us":    float(np.max(times_us)),
        "n_reps":            len(times_us),
    }


# ---------------------------------------------------------------------------
# Pareto
# ---------------------------------------------------------------------------

def build_pareto(latency_df, results_root, gran, out_dir):
    """
    Join latency results with mean WMAPE from grand_summary.csv for
    each (exp_type, platform, suite, feature_set) combination.
    """
    rows = []
    for _, r in latency_df.iterrows():
        # grand_summary.csv sits under .../platform/suite/feature_set/
        # (one level above the direction directories)
        pattern = os.path.join(
            results_root, r["exp_type"], r["platform"], "*",
            r["suite"], r["feature_set"], "grand_summary.csv",
        )
        matches = glob.glob(pattern)
        if not matches:
            # try without the cpu/direction level (arm, x86_server layout)
            pattern2 = os.path.join(
                results_root, r["exp_type"], r["platform"],
                r["suite"], r["feature_set"], "grand_summary.csv",
            )
            matches = glob.glob(pattern2)
        if not matches:
            continue
        gs = pd.read_csv(matches[0])
        rows.append({
            "exp_type":        r["exp_type"],
            "platform":        r["platform"],
            "suite":           r["suite"],
            "feature_set":     r["feature_set"],
            "n_features":      r["n_features"],
            "n_trees":         r["n_trees"],
            "wmape_ml":        gs["wmape_ml"].mean(),
            "mape_ml":         gs["mape_ml"].mean(),
            "latency_mean_us": r["latency_mean_us"],
            "latency_p99_us":  r["latency_p99_us"],
        })

    if not rows:
        print("  [INFO] No grand_summary.csv files found — skipping pareto CSV.")
        return

    pareto_df = pd.DataFrame(rows).sort_values(["exp_type", "platform", "suite", "feature_set"])
    out_path  = os.path.join(out_dir, f"pareto_{gran}.csv")
    pareto_df.to_csv(out_path, index=False)
    print(f"\nPareto table → {out_path}")
    print(pareto_df.to_string(index=False))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gran", default="10M",
                        help="Instruction-window granularity to benchmark (default: 10M).")
    parser.add_argument("--reps", type=int, default=0,
                        help="Fixed rep count. 0 = adaptive (default).")
    parser.add_argument("--warmup", type=int, default=WARMUP_REPS,
                        help=f"Warmup reps before timing (default: {WARMUP_REPS}).")
    parser.add_argument("--cpu", type=int, default=0,
                        help="CPU core to pin to (default: 0).")
    parser.add_argument("--results_root", default="results/cross_platform",
                        help="Root directory containing cross_proc/ and cross_freq/.")
    parser.add_argument("--out_dir", default="results/cross_platform/latency",
                        help="Directory to write output CSVs to.")
    args = parser.parse_args()

    pin_cpu(args.cpu)

    entries = discover_models(args.results_root, args.gran, FEATURE_SETS)
    if not entries:
        print("No trained models found — check --results_root and --gran.")
        return

    print(f"\nFound {len(entries)} (exp_type × platform × suite × feature_set) groups "
          f"at granularity {args.gran}\n")

    os.makedirs(args.out_dir, exist_ok=True)
    rows = []

    for entry in entries:
        label = (f"{entry['exp_type']}/{entry['platform']}/"
                 f"{entry['suite']}/{entry['feature_set']}")
        print(f"[{label}]  {os.path.basename(entry['model_path'])}")

        model = CatBoostRegressor()
        model.load_model(entry["model_path"])

        X    = build_input(model)
        reps = args.reps if args.reps > 0 else adaptive_reps(model, X, args.warmup)
        times_us = time_model(model, X, reps, args.warmup)
        stats    = summarize(times_us)

        n_feat  = len(model.feature_names_)
        n_trees = model.tree_count_
        print(f"  n_features={n_feat}  n_trees={n_trees}  reps={reps:,}  "
              f"mean={stats['latency_mean_us']:.2f}µs  "
              f"p50={stats['latency_p50_us']:.2f}µs  "
              f"p99={stats['latency_p99_us']:.2f}µs")

        row = {
            "exp_type":    entry["exp_type"],
            "platform":    entry["platform"],
            "suite":       entry["suite"],
            "feature_set": entry["feature_set"],
            "n_features":  n_feat,
            "n_trees":     n_trees,
            "model_path":  entry["model_path"],
        }
        row.update(stats)
        rows.append(row)

    out_df   = pd.DataFrame(rows)
    out_path = os.path.join(args.out_dir, f"inference_latency_{args.gran}.csv")
    out_df.to_csv(out_path, index=False)
    print(f"\nResults → {out_path}")

    build_pareto(out_df, args.results_root, args.gran, args.out_dir)


if __name__ == "__main__":
    main()

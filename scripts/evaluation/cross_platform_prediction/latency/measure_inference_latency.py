#!/usr/bin/env python3
"""
Measure single-sample inference latency for trained cross-platform CatBoost
models (cross-processor and cross-frequency, full/top2/top4 feature sets).

For each (exp_type, group, suite, feature_set) combination, picks one
representative trained model, builds a synthetic single-row input matching
its feature schema, and times ~10,000 single-sample model.predict() calls.
Writes a summary CSV and (if grand_summary.csv files are available) a
latency-vs-accuracy pareto CSV.
"""

import argparse
import glob
import os
import time

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


EXP_TYPES = ["cross_proc", "cross_freq"]
FEATURE_SETS = ["full", "top2", "top4"]


def discover_models(results_root, platform_dir):
    """
    Find one representative trained model per
    (exp_type, group, suite, feature_set), where `group` is the
    direction (cross_proc, e.g. cpu0_to_cpu16) or target cpu (cross_freq,
    e.g. cpu0). Returns a list of dicts, sorted, or [] if nothing found.
    """
    groups = {}
    for exp_type in EXP_TYPES:
        for feature_set in FEATURE_SETS:
            pattern = os.path.join(
                results_root, exp_type, platform_dir, "*", "*", feature_set, "**", "model_*.cbm"
            )
            base = os.path.join(results_root, exp_type, platform_dir)
            for path in sorted(glob.glob(pattern, recursive=True)):
                rel = os.path.relpath(path, base)
                group_dir, suite = rel.split(os.sep)[:2]
                key = (exp_type, group_dir, suite, feature_set)
                groups.setdefault(key, path)

    for fs in FEATURE_SETS:
        if not any(k[3] == fs for k in groups):
            print(f"  [SKIP] No '{fs}' models found yet, skipping that feature set.")

    return [
        {"exp_type": k[0], "group": k[1], "suite": k[2], "feature_set": k[3], "model_path": v}
        for k, v in sorted(groups.items())
    ]


def build_synthetic_input(model):
    """Build a single-row DataFrame matching the model's feature schema.
    Timing is value-independent; only column count/dtypes/cat-feature
    config affect predict() cost."""
    feature_names = model.feature_names_
    cat_indices = set(model.get_cat_feature_indices())
    data = {}
    for i, name in enumerate(feature_names):
        data[name] = ["Compute_Bound"] if i in cat_indices else [1.0]
    return pd.DataFrame(data, columns=feature_names)


def time_model(model, X, reps, warmup):
    for _ in range(warmup):
        model.predict(X, thread_count=1)

    times_us = np.empty(reps)
    for i in range(reps):
        start = time.perf_counter()
        model.predict(X, thread_count=1)
        times_us[i] = (time.perf_counter() - start) * 1e6
    return times_us


def summarize(times_us):
    return {
        "latency_mean_us": np.mean(times_us),
        "latency_std_us": np.std(times_us),
        "latency_median_us": np.median(times_us),
        "latency_p50_us": np.percentile(times_us, 50),
        "latency_p90_us": np.percentile(times_us, 90),
        "latency_p95_us": np.percentile(times_us, 95),
        "latency_p99_us": np.percentile(times_us, 99),
        "latency_p999_us": np.percentile(times_us, 99.9),
        "latency_min_us": np.min(times_us),
        "latency_max_us": np.max(times_us),
    }


def build_pareto(latency_df, results_root, platform_dir, out_dir):
    rows = []
    for _, r in latency_df.iterrows():
        gs_path = os.path.join(
            results_root, r["exp_type"], platform_dir, r["group"], r["suite"],
            r["feature_set"], "grand_summary.csv",
        )
        if not os.path.exists(gs_path):
            continue
        gs = pd.read_csv(gs_path)
        rows.append({
            "exp_type": r["exp_type"],
            "group": r["group"],
            "suite": r["suite"],
            "feature_set": r["feature_set"],
            "n_features": r["n_features"],
            "n_cat_features": r["n_cat_features"],
            "wmape_ml": gs["wmape_ml"].mean(),
            "latency_mean_us": r["latency_mean_us"],
            "latency_p99_us": r["latency_p99_us"],
        })

    if not rows:
        print("  [SKIP] No grand_summary.csv files found, skipping pareto CSV.")
        return

    pareto_df = pd.DataFrame(rows)
    out_path = os.path.join(out_dir, f"pareto_{platform_dir}.csv")
    pareto_df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=10000,
                         help="Number of single-sample predict() calls to time per model.")
    parser.add_argument("--warmup", type=int, default=100,
                         help="Number of untimed warmup predict() calls per model.")
    parser.add_argument("--results_root", default="results/cross_platform",
                         help="Root directory containing cross_proc/ and cross_freq/ results.")
    parser.add_argument("--out_dir", default="results/cross_platform/latency",
                         help="Directory to write output CSVs to.")
    parser.add_argument("--platform_dir", default="x86_10M",
                         help="Platform/granularity subdirectory to scan (e.g. x86_10M).")
    args = parser.parse_args()

    models = discover_models(args.results_root, args.platform_dir)
    if not models:
        print("No trained models found.")
        return

    os.makedirs(args.out_dir, exist_ok=True)

    rows = []
    for entry in models:
        print(f"Timing {entry['exp_type']}/{entry['group']}/{entry['suite']}/{entry['feature_set']} "
              f"({os.path.basename(entry['model_path'])}) ...")

        model = CatBoostRegressor()
        model.load_model(entry["model_path"])

        X = build_synthetic_input(model)
        times_us = time_model(model, X, args.reps, args.warmup)
        stats = summarize(times_us)

        row = {
            "exp_type": entry["exp_type"],
            "group": entry["group"],
            "suite": entry["suite"],
            "feature_set": entry["feature_set"],
            "n_features": len(model.feature_names_),
            "n_cat_features": len(model.get_cat_feature_indices()),
            "model_path": entry["model_path"],
        }
        row.update(stats)
        rows.append(row)
        print(f"    n_features={row['n_features']}  mean={stats['latency_mean_us']:.2f}us  "
              f"p99={stats['latency_p99_us']:.2f}us")

    out_df = pd.DataFrame(rows)
    out_path = os.path.join(args.out_dir, f"inference_latency_{args.platform_dir}.csv")
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")

    build_pareto(out_df, args.results_root, args.platform_dir, args.out_dir)


if __name__ == "__main__":
    main()

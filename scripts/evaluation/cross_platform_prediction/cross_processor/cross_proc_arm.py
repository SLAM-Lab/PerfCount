"""
cross_proc_arm.py
=================
Cross-processor CPU-cycle prediction for ARM heterogeneous platforms
(e.g. In-Order cpu1 <-> Out-of-Order cpu4 on ARM edge).

Uses the unified feature engineering and CatBoost model from shared_features.py,
keeping the model identical to cross_freq_arm.py.

Typical usage
-------------
# InO -> OOO, all common frequencies
python cross_proc_arm.py \\
    --data_dir ../../../../processed_data_100M/arm_edge_heterogeneous \\
    --out_dir  ../../../../results/cross_platform/cross_processor/arm_edge_100M \\
    --src_cpu 1 --tgt_cpu 4

# DaCapo benchmarks only
python cross_proc_arm.py \\
    --data_dir ../../../../processed_data_100M/arm_edge_heterogeneous \\
    --out_dir  ../../../../results/cross_platform/cross_processor/arm_edge_100M \\
    --src_cpu 1 --tgt_cpu 4 --suite dacapo

# SPEC benchmarks only
python cross_proc_arm.py \\
    --data_dir ../../../../processed_data_100M/arm_edge_heterogeneous \\
    --out_dir  ../../../../results/cross_platform/cross_processor/arm_edge_100M \\
    --src_cpu 1 --tgt_cpu 4 --suite spec

# OOO -> InO, strict LOOCV
python cross_proc_arm.py \\
    --data_dir ../../../../processed_data_100M/arm_edge_heterogeneous \\
    --out_dir  ../../../../results/cross_platform/cross_processor/arm_edge_100M \\
    --src_cpu 4 --tgt_cpu 1 --strict_loocv
"""

import os
import sys
import argparse
import glob
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from shared_features import (
    add_feature_args,
    build_features,
    restrict_input_counters,
    build_model,
    cat_feature_names,
    prepare_bench_df,
    compute_metrics,
    load_fold_if_done,
    try_load_model,
    run_loocv,
    print_summary,
    save_feature_importance,
)


# =============================================================================
# 1.  WORKER  (one LOOCV fold)
# =============================================================================

def process_fold(test_bench, train_dfs, test_df, args, freq_ratio=1.0, out_dir="."):
    try:
        cached = load_fold_if_done(out_dir, test_bench, freq_ratio)
        if cached is not None:
            return cached

        train_full = pd.concat(train_dfs, ignore_index=True)

        train_full = restrict_input_counters(train_full, "_src", getattr(args, "input_counters", None))
        test_df    = restrict_input_counters(test_df, "_src", getattr(args, "input_counters", None))

        X_train = build_features(train_full, suffix="_src", args=args)
        X_test  = build_features(test_df,    suffix="_src", args=args)

        if X_train.empty or X_test.empty:
            return None

        for c in X_train.columns:
            if c not in X_test.columns:
                X_test[c] = 0
        X_test = X_test[X_train.columns]

        src_clean   = train_full["source_val"].replace(0, np.nan).fillna(1e-9)
        ratio_train = train_full["target_y"] / src_clean
        y_train_log = np.log(np.clip(ratio_train, 0.05, 50.0))

        src_clean_t = test_df["source_val"].replace(0, np.nan).fillna(1e-9)
        ratio_test  = test_df["target_y"] / src_clean_t
        y_test_log  = np.log(np.clip(ratio_test, 0.05, 50.0))

        cat_feats = cat_feature_names(args)
        model = try_load_model(out_dir, test_bench)
        if model is None:
            model = build_model(cat_feats)
            model.fit(
                X_train, y_train_log,
                eval_set=(X_test, y_test_log),
                early_stopping_rounds=200,
            )

        importances = dict(zip(X_train.columns.tolist(), model.get_feature_importance()))

        pred_log    = model.predict(X_test)
        pred_ratio  = np.exp(pred_log)
        pred_cycles = pred_ratio * test_df["source_val"].values

        y_true_cycles = test_df["target_y"].values
        src_cycles    = test_df["source_val"].values

        m_ml    = compute_metrics(y_true_cycles, pred_cycles)
        m_copy  = compute_metrics(y_true_cycles, src_cycles)
        m_scale = compute_metrics(y_true_cycles, src_cycles * freq_ratio)

        os.makedirs(out_dir, exist_ok=True)
        model.save_model(os.path.join(out_dir, f"model_{test_bench}.cbm"))
        pd.DataFrame({
            "source_val":       src_cycles,
            "target_actual":    y_true_cycles,
            "target_predicted": pred_cycles,
            "ratio_actual":     test_df["target_y"].values / src_clean_t.values,
            "ratio_predicted":  pred_ratio,
        }).to_csv(os.path.join(out_dir, f"predictions_{test_bench}.csv"), index=False)

        return {
            "bench":               test_bench,
            "wmape":               m_ml["wmape"],
            "mape":                m_ml["mape"],
            "mdape":               m_ml["mdape"],
            "wmape_copy":          m_copy["wmape"],
            "mape_copy":           m_copy["mape"],
            "wmape_scale":         m_scale["wmape"],
            "mape_scale":          m_scale["mape"],
            "feature_importances": importances,
        }

    except Exception as e:
        print(f"  [WARN] Fold '{test_bench}' failed: {e}")
        return None


# =============================================================================
# 2.  DATA LOADING
# =============================================================================

def _resolve_suite_dirs(data_dir, suite):
    """Return list of suite subdirectories to scan based on --suite flag."""
    candidates = sorted(
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    )
    if not candidates:
        return [data_dir]
    if suite != "all":
        return [os.path.join(data_dir, d) for d in candidates if d == suite]
    has_c1 = "dacapo_c1" in candidates
    dirs = []
    for d in candidates:
        if d == "dacapo_c2" and has_c1:
            continue
        dirs.append(os.path.join(data_dir, d))
    return dirs


def load_cpu_data(data_dir, cpu_id, suite="all"):
    """
    Load all aligned_*.csv files for a given cpu_id.

    Returns
    -------
    data_map : dict[(freq_str, bench_phase_str) -> pd.DataFrame]
    """
    data_map = {}
    if not os.path.exists(data_dir):
        print(f"[ERROR] Directory not found: {os.path.abspath(data_dir)}")
        return data_map

    pattern = re.compile(
        r"aligned_(?P<bench>.+?)_(?P<freq>[\d.]+)GHz"
        r"_cpu(?P<cpu>\d+)_phase(?P<phase>\d+)\.csv"
    )

    for suite_dir in _resolve_suite_dirs(data_dir, suite):
        for fpath in glob.glob(os.path.join(suite_dir, "**", "aligned_*.csv"), recursive=True):
            fname = os.path.basename(fpath)
            m = pattern.match(fname)
            if not m or m.group("cpu") != str(cpu_id):
                continue
            try:
                df = pd.read_csv(fpath)
                df.columns = [c.strip() for c in df.columns]
                if "instructions" in df.columns and "cpu_cycles" in df.columns:
                    df = df[(df["instructions"] > 100_000) & (df["cpu_cycles"] > 0)]
                if not df.empty:
                    key = (m.group("freq"), f"{m.group('bench')}_phase{m.group('phase')}")
                    data_map[key] = df
            except Exception as e:
                print(f"  [WARN] Could not load {fname}: {e}")

    return data_map


# =============================================================================
# 3.  FREQUENCY-PAIR RUNNER
# =============================================================================

def run_freq_pair(src_map, tgt_map, src_freq, tgt_freq, src_cpu, tgt_cpu, args, out_dir):
    """
    LOOCV for one (src_cpu@src_freq -> tgt_cpu@tgt_freq) combination.
    """
    benches_src = {k[1] for k in src_map if k[0] == src_freq}
    benches_tgt = {k[1] for k in tgt_map if k[0] == tgt_freq}
    common = sorted(benches_src & benches_tgt)

    if len(common) < 2:
        print(f"  Skipping cpu{src_cpu}@{src_freq}->cpu{tgt_cpu}@{tgt_freq}: "
              f"only {len(common)} common workload(s).")
        return None

    freq_ratio = float(tgt_freq) / float(src_freq) if float(src_freq) > 0 else 1.0
    direction_dir = os.path.join(
        out_dir, f"cpu{src_cpu}_{src_freq}GHz_to_cpu{tgt_cpu}_{tgt_freq}GHz"
    )
    os.makedirs(direction_dir, exist_ok=True)

    bench_dfs = {}
    for b in common:
        merged = prepare_bench_df(
            src_map[(src_freq, b)].copy(),
            tgt_map[(tgt_freq, b)].copy(),
            target_key="cpu_cycles",
        )
        if merged is not None:
            bench_dfs[b] = merged

    if len(bench_dfs) < 2:
        print(f"  Skipping cpu{src_cpu}@{src_freq}->cpu{tgt_cpu}@{tgt_freq}: "
              f"only {len(bench_dfs)} valid pairs after filtering.")
        return None

    results = run_loocv(
        bench_dfs,
        process_fold,
        args,
        extra_kwargs={"freq_ratio": freq_ratio, "out_dir": direction_dir},
    )

    if not results:
        return None

    label = f"cpu{src_cpu} {src_freq} GHz -> cpu{tgt_cpu} {tgt_freq} GHz"
    df_res = print_summary(results, label=label)
    df_res.to_csv(os.path.join(direction_dir, "per_fold_results.csv"), index=False)
    save_feature_importance(results, direction_dir)

    return {
        "pair":        f"cpu{src_cpu}@{src_freq}->cpu{tgt_cpu}@{tgt_freq}",
        "wmape_ml":    df_res["wmape"].mean(),
        "mape_ml":     df_res["mape"].mean(),
        "wmape_copy":  df_res["wmape_copy"].mean(),
        "mape_copy":   df_res["mape_copy"].mean(),
        "wmape_scale": df_res["wmape_scale"].mean(),
        "mape_scale":  df_res["mape_scale"].mean(),
        "n_folds":     len(df_res),
    }


# =============================================================================
# 4.  MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Cross-processor prediction for ARM heterogeneous platforms."
    )
    parser.add_argument("--data_dir", required=True,
                        help="Directory containing aligned ARM CSV files")
    parser.add_argument("--out_dir", required=True,
                        help="Output directory for results and diagnostics")
    parser.add_argument("--src_cpu", required=True, type=str,
                        help="Source CPU ID (e.g. '1' for In-Order)")
    parser.add_argument("--tgt_cpu", required=True, type=str,
                        help="Target CPU ID (e.g. '4' for Out-of-Order)")
    parser.add_argument("--suite", choices=["all", "spec_2017", "spec_2026", "dacapo_c2", "dacapo_c1"], default="all",
                        help="Benchmark suite to include in LOOCV folds: "
                             "'all' (default), 'spec_2017', 'spec_2026', 'dacapo_c2', or 'dacapo_c1'.")
    add_feature_args(parser)
    parser.add_argument("--force", action="store_true", default=False,
                        help="Re-run even if grand_summary.csv already exists in --out_dir.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    summary_path = os.path.join(args.out_dir, "grand_summary.csv")
    if os.path.exists(summary_path) and not args.force:
        print(f"[SKIP] {summary_path} already exists. Use --force to re-run.")
        return
    print(f"\n{'='*60}")
    print(f"  cross_proc_arm | data: {os.path.abspath(args.data_dir)}")
    print(f"  cpu{args.src_cpu} -> cpu{args.tgt_cpu}")
    print(f"  Features       | raw counters")
    print(f"  LOOCV          | strict={args.strict_loocv}  jobs={args.jobs}  suite={args.suite}")
    print(f"{'='*60}\n")

    src_map = load_cpu_data(args.data_dir, args.src_cpu, suite=args.suite)
    tgt_map = load_cpu_data(args.data_dir, args.tgt_cpu, suite=args.suite)

    if not src_map:
        print(f"[ERROR] No data found for src cpu{args.src_cpu}. Check --data_dir.")
        return
    if not tgt_map:
        print(f"[ERROR] No data found for tgt cpu{args.tgt_cpu}. Check --data_dir.")
        return

    src_freqs = sorted({k[0] for k in src_map})
    tgt_freqs = sorted({k[0] for k in tgt_map})
    print(f"  cpu{args.src_cpu} frequencies : {src_freqs}")
    print(f"  cpu{args.tgt_cpu} frequencies : {tgt_freqs}")

    all_summary = []
    for sf in src_freqs:
        for tf in tgt_freqs:
            print(f"\n  --- cpu{args.src_cpu} {sf} GHz  ->  cpu{args.tgt_cpu} {tf} GHz ---")
            try:
                res = run_freq_pair(
                    src_map, tgt_map, sf, tf,
                    args.src_cpu, args.tgt_cpu, args, args.out_dir,
                )
            except Exception as e:
                print(f"  [ERROR] Pair cpu{args.src_cpu}@{sf}->cpu{args.tgt_cpu}@{tf} failed: {e}")
                res = None
            if res:
                all_summary.append(res)

    if all_summary:
        df_summary = pd.DataFrame(all_summary)
        summary_path = os.path.join(args.out_dir, "grand_summary.csv")
        df_summary.to_csv(summary_path, index=False)
        print(f"\n{'='*60}")
        print("  GRAND SUMMARY")
        print("=" * 60)
        print(df_summary.to_string(index=False))
        print(f"\n  Saved to: {summary_path}")


if __name__ == "__main__":
    main()

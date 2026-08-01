"""
cross_freq_x86.py
=================
Cross-frequency CPU-cycle prediction for x86 platforms (desktop heterogeneous).

Uses raw hardware counters and CatBoost model from shared_features.py.

Typical usage
-------------
python cross_freq_x86.py --data_dir path/to/x86_desktop_heterogeneous --out_dir results/cf_x86 \\
       --suite spec_2026 --target_cpu 0
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
    run_general_model,
    print_summary,
    save_feature_importance,
    filter_excluded_benchmarks,
    RATIO_CLIP_LO,
    RATIO_CLIP_HI,
)


# =============================================================================
# 1.  WORKER  (one LOOCV fold)
# =============================================================================

def process_fold(test_bench, train_dfs, test_df, args, freq_ratio=1.0, out_dir="."):
    """
    Train on all-but-one workload, predict the held-out workload.

    Parameters
    ----------
    test_bench  : str            — workload identifier (for diagnostics)
    train_dfs   : list[DataFrame]
    test_df     : DataFrame
    args        : argparse.Namespace
    freq_ratio  : float          — tgt_freq / src_freq (used for scale baseline)
    out_dir     : str            — where to write per-fold CSVs

    Returns
    -------
    dict with bench, mape, mdape, mape_copy, mape_scale
    """
    try:
        cached = load_fold_if_done(out_dir, test_bench, freq_ratio)
        if cached is not None:
            return cached

        train_full = pd.concat(train_dfs, ignore_index=True)

        # Per-workload equal weighting: each workload contributes total weight 1
        # regardless of trace length, so short workloads are not drowned out.
        if getattr(args, "equal_weight", False):
            sample_weights = np.concatenate(
                [np.full(len(df), 1.0 / len(df)) for df in train_dfs]
            )
        else:
            sample_weights = None

        train_full = restrict_input_counters(train_full, "_src", getattr(args, "input_counters", None))
        test_df    = restrict_input_counters(test_df, "_src", getattr(args, "input_counters", None))

        X_train = build_features(train_full, suffix="_src", args=args)
        X_test  = build_features(test_df,    suffix="_src", args=args)

        if X_train.empty or X_test.empty:
            return None

        common_cols = sorted(set(X_train.columns) & set(X_test.columns))
        X_train = X_train[common_cols]
        X_test = X_test[common_cols]

        src_clean   = train_full["source_val"].replace(0, np.nan).fillna(1e-9)
        ratio_train = train_full["target_y"] / src_clean
        y_train_log = np.log(np.clip(ratio_train, RATIO_CLIP_LO, RATIO_CLIP_HI))

        src_clean_t = test_df["source_val"].replace(0, np.nan).fillna(1e-9)
        ratio_test  = test_df["target_y"] / src_clean_t
        y_test_log  = np.log(np.clip(ratio_test, RATIO_CLIP_LO, RATIO_CLIP_HI))

        cat_feats = cat_feature_names(args)
        model = try_load_model(out_dir, test_bench)
        if model is None:
            model = build_model(cat_feats)
            model.fit(
                X_train, y_train_log,
                eval_set=(X_test, y_test_log),
                early_stopping_rounds=200,
                sample_weight=sample_weights,
            )

        importances = dict(zip(X_train.columns.tolist(), model.get_feature_importance()))

        # Predictions
        pred_log    = model.predict(X_test)
        pred_ratio  = np.exp(pred_log)
        pred_cycles = pred_ratio * test_df["source_val"].values

        y_true_cycles = test_df["target_y"].values
        src_cycles    = test_df["source_val"].values

        m_ml    = compute_metrics(y_true_cycles, pred_cycles)
        m_copy  = compute_metrics(y_true_cycles, src_cycles)
        # ref_cycles ~ wall time at fixed ref freq; assuming constant cpu_cycles
        # work, wall time scales with src_freq/tgt_freq (inverse of freq_ratio).
        m_scale = compute_metrics(y_true_cycles, src_cycles / freq_ratio)

        os.makedirs(out_dir, exist_ok=True)
        model.save_model(os.path.join(out_dir, f"model_{test_bench}.cbm"))
        if getattr(args, "save_predictions", False):
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


def load_x86_data(data_dir, target_cpu=None, suite="all"):
    """
    Scan data_dir for aligned_*.csv files following x86 naming convention:
        aligned_<bench>_<freq>GHz[_cpu<N>]_phase<P>.csv

    For x86 desktop heterogeneous: cpu0 = P-cores, cpu16 = E-cores.

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
        r"(?:_cpu(?P<cpu>\d+))?_phase(?P<phase>\d+)\.csv"
    )

    for suite_dir in _resolve_suite_dirs(data_dir, suite):
        for fpath in glob.glob(os.path.join(suite_dir, "**", "aligned_*.csv"), recursive=True):
            fname = os.path.basename(fpath)
            m = pattern.match(fname)
            if not m:
                continue
            if target_cpu is not None and m.group("cpu") != str(target_cpu):
                continue
            try:
                df = pd.read_csv(fpath)
                df.columns = [c.strip() for c in df.columns]
                if "instructions" in df.columns and "ref_cycles" in df.columns:
                    df = df[(df["instructions"] > 100_000) & (df["ref_cycles"] > 0)]
                if not df.empty:
                    key = (m.group("freq"), f"{m.group('bench')}_phase{m.group('phase')}")
                    data_map[key] = df
            except Exception as e:
                print(f"  [WARN] Could not load {fname}: {e}")

    return data_map


# =============================================================================
# 3.  FREQUENCY-PAIR RUNNER
# =============================================================================

def run_freq_pair(data_map, src_freq, tgt_freq, args, out_dir):
    """
    Build and evaluate LOOCV for one (src_freq -> tgt_freq) direction.
    """
    benches_at_src = {k[1] for k in data_map if k[0] == src_freq}
    benches_at_tgt = {k[1] for k in data_map if k[0] == tgt_freq}
    common = sorted(benches_at_src & benches_at_tgt)

    if len(common) < 2:
        print(f"  Skipping {src_freq}->{tgt_freq}: only {len(common)} common workload(s).")
        return None

    freq_ratio = float(tgt_freq) / float(src_freq) if float(src_freq) > 0 else 1.0
    direction_dir = os.path.join(out_dir, f"{src_freq}GHz_to_{tgt_freq}GHz")
    os.makedirs(direction_dir, exist_ok=True)

    bench_dfs = {}
    for b in common:
        merged = prepare_bench_df(
            data_map[(src_freq, b)].copy(),
            data_map[(tgt_freq, b)].copy(),
            target_key=getattr(args, "target_key", "cpu_cycles"),
        )
        if merged is not None:
            bench_dfs[b] = merged

    if len(bench_dfs) < 2:
        print(f"  Skipping {src_freq}->{tgt_freq}: only {len(bench_dfs)} valid pairs after filtering.")
        return None

    if getattr(args, "mode", "loocv") == "loocv":
        results = run_loocv(
            bench_dfs,
            process_fold,
            args,
            extra_kwargs={"freq_ratio": freq_ratio, "out_dir": direction_dir},
        )
    else:
        results = run_general_model(bench_dfs, args, freq_ratio, direction_dir, args.mode)

    if not results:
        return None

    df_res = print_summary(results, label=f"{src_freq} GHz → {tgt_freq} GHz")
    df_res.to_csv(os.path.join(direction_dir, "per_fold_results.csv"), index=False)
    save_feature_importance(results, direction_dir)

    return {
        "pair":        f"{src_freq}->{tgt_freq}",
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
        description="Cross-frequency prediction for x86 platforms."
    )
    parser.add_argument("--data_dir", required=True,
                        help="Directory containing aligned x86 CSV files")
    parser.add_argument("--out_dir", required=True,
                        help="Output directory for results and diagnostics")
    parser.add_argument("--target_cpu", type=str, default=None,
                        help="Filter to a specific CPU ID (e.g. '0' for P-cores, '16' for E-cores). "
                             "Leave unset to use all CPUs in the directory.")
    parser.add_argument("--suite", choices=["all", "spec_2017", "spec_2026", "dacapo_c2", "dacapo_c1"], default="all",
                        help="Benchmark suite to include in LOOCV folds: "
                             "'all' (default), 'spec_2017', 'spec_2026', 'dacapo_c2', or 'dacapo_c1'.")
    parser.add_argument("--save_predictions", action="store_true", default=False,
                        help="Save per-fold predictions_{bench}.csv files (disabled by default).")
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
    print(f"  cross_freq_x86 | data: {os.path.abspath(args.data_dir)}")
    print(f"  Features       | raw counters")
    print(f"  LOOCV          | strict={args.strict_loocv}  jobs={args.jobs}  suite={args.suite}  equal_weight={args.equal_weight}")
    print(f"{'='*60}\n")

    data_map = load_x86_data(args.data_dir, target_cpu=args.target_cpu, suite=args.suite)
    data_map = filter_excluded_benchmarks(data_map, args)
    if not data_map:
        print("[ERROR] No data loaded. Check --data_dir and file naming.")
        return

    freqs = sorted({k[0] for k in data_map})
    print(f"  Found {len(data_map)} benchmark files across frequencies: {freqs}")

    all_summary = []
    for src in freqs:
        for tgt in freqs:
            if src == tgt:
                continue
            print(f"\n  --- {src} GHz  →  {tgt} GHz ---")
            try:
                res = run_freq_pair(data_map, src, tgt, args, args.out_dir)
            except Exception as e:
                print(f"  [ERROR] Pair {src}->{tgt} failed: {e}")
                res = None
            if res:
                all_summary.append(res)

    if all_summary:
        df_summary = pd.DataFrame(all_summary)
        summary_path = os.path.join(args.out_dir, "grand_summary.csv")
        df_summary.to_csv(summary_path, index=False)
        print(f"\n{'='*60}")
        print("  GRAND SUMMARY")
        print("="*60)
        print(df_summary.to_string(index=False))
        print(f"\n  Saved to: {summary_path}")


if __name__ == "__main__":
    main()

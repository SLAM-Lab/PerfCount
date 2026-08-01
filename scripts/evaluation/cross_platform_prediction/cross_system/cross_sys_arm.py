"""
cross_sys_arm.py
================
Cross-system CPU-cycle prediction across ARM platforms
(arm_server, arm_desktop, arm_edge_heterogeneous).

Uses the unified feature engineering and CatBoost model from shared_features.py,
keeping the model identical to cross_freq_arm.py and cross_proc_arm.py.

Workloads common to both platforms are discovered automatically.
Use --suite to control which benchmark suites are included:
  spec  — SPEC workloads only (default; valid for all platform pairs)
  dacapo — DaCapo workloads only (server<->desktop only)
  all   — all common workloads across both suites

File naming conventions supported
----------------------------------
With CPU tag (most platforms):
    aligned_<bench>_<freq>GHz_cpu<N>_phase<P>.csv
Without CPU tag (arm_server 1000M):
    aligned_<bench>_<freq>GHz_phase<P>.csv

Typical usage
-------------
# ARM server -> ARM desktop, all common frequencies
python cross_sys_arm.py \\
    --src_data_dir  ../../../../processed_data_100M/arm_server \\
    --tgt_data_dir  ../../../../processed_data_100M/arm_desktop \\
    --out_dir       ../../../../results/cross_platform/cross_system/arm/100M/server_to_desktop

# ARM server -> ARM edge OoO core
python cross_sys_arm.py \\
    --src_data_dir  ../../../../processed_data_100M/arm_server \\
    --tgt_data_dir  ../../../../processed_data_100M/arm_edge_heterogeneous \\
    --tgt_cpu 4 \\
    --out_dir       ../../../../results/cross_platform/cross_system/arm/100M/server_to_edge_ooo
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

# Matches both naming conventions:
#   aligned_<bench>_<freq>GHz_cpu<N>_phase<P>.csv
#   aligned_<bench>_<freq>GHz_phase<P>.csv
_FILE_PATTERN = re.compile(
    r"aligned_(?P<bench>.+?)_(?P<freq>[\d.]+)GHz"
    r"(?:_cpu(?P<cpu>\d+))?_phase(?P<phase>\d+)\.csv"
)


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


def load_platform_data(data_dir, cpu_id=None, suite="all"):
    """
    Load all aligned_*.csv files from data_dir.

    Parameters
    ----------
    data_dir : str
    cpu_id   : str or None
        If given, only load files whose cpu tag matches (e.g. '0', '4').
        If None, load all files regardless of cpu tag.
    suite    : str
        Suite directory to scan ('all', 'spec_2017', 'spec_2026', 'dacapo_c2', 'dacapo_c1').

    Returns
    -------
    data_map : dict[(freq_str, bench_phase_str) -> pd.DataFrame]
    """
    data_map = {}
    if not os.path.exists(data_dir):
        print(f"[ERROR] Directory not found: {os.path.abspath(data_dir)}")
        return data_map

    for suite_dir in _resolve_suite_dirs(data_dir, suite):
        for fpath in glob.glob(os.path.join(suite_dir, "**", "aligned_*.csv"), recursive=True):
            fname = os.path.basename(fpath)
            m = _FILE_PATTERN.match(fname)
            if not m:
                continue

            file_cpu = m.group("cpu")

            if cpu_id is not None and file_cpu != str(cpu_id):
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

def run_freq_pair(src_map, tgt_map, src_freq, tgt_freq, src_label, tgt_label,
                  args, out_dir):
    """
    LOOCV for one (src_platform@src_freq -> tgt_platform@tgt_freq) combination.
    Only workloads present in both maps at the given frequencies are used.
    """
    benches_src = {k[1] for k in src_map if k[0] == src_freq}
    benches_tgt = {k[1] for k in tgt_map if k[0] == tgt_freq}
    common = sorted(benches_src & benches_tgt)

    if len(common) < 2:
        print(f"  Skipping {src_label}@{src_freq}->{tgt_label}@{tgt_freq}: "
              f"only {len(common)} common workload(s).")
        return None

    freq_ratio = float(tgt_freq) / float(src_freq) if float(src_freq) > 0 else 1.0
    direction_dir = os.path.join(
        out_dir, f"{src_label}_{src_freq}GHz_to_{tgt_label}_{tgt_freq}GHz"
    )
    os.makedirs(direction_dir, exist_ok=True)

    bench_dfs = {}
    for b in common:
        merged = prepare_bench_df(
            src_map[(src_freq, b)].copy(),
            tgt_map[(tgt_freq, b)].copy(),
            target_key=args.target_key,
        )
        if merged is not None:
            bench_dfs[b] = merged

    if len(bench_dfs) < 2:
        print(f"  Skipping {src_label}@{src_freq}->{tgt_label}@{tgt_freq}: "
              f"only {len(bench_dfs)} valid pairs after filtering.")
        return None

    if getattr(args, "mode", "loocv") == "loocv":
        results = run_loocv(
            bench_dfs,
            process_fold,
            args,
            extra_kwargs={"freq_ratio": freq_ratio, "out_dir": direction_dir},
        )
    else:
        # scale_mode="multiply" matches this trainer's process_fold baseline
        # (src_cycles * freq_ratio), unlike cross_freq/cross_proc which divide.
        results = run_general_model(bench_dfs, args, freq_ratio, direction_dir,
                                    args.mode, scale_mode="multiply")

    if not results:
        return None

    label = f"{src_label} {src_freq} GHz -> {tgt_label} {tgt_freq} GHz"
    df_res = print_summary(results, label=label)
    df_res.to_csv(os.path.join(direction_dir, "per_fold_results.csv"), index=False)
    save_feature_importance(results, direction_dir)

    return {
        "pair":       f"{src_label}@{src_freq}->{tgt_label}@{tgt_freq}",
        "n_common":   len(common),
        "n_folds":    len(df_res),
        "mape_ml":    df_res["mape"].mean(),
        "mape_copy":  df_res["mape_copy"].mean(),
        "mape_scale": df_res["mape_scale"].mean(),
    }


# =============================================================================
# 4.  MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Cross-system ARM prediction using shared_features model."
    )
    parser.add_argument("--src_data_dir", required=True,
                        help="Source platform data directory")
    parser.add_argument("--tgt_data_dir", required=True,
                        help="Target platform data directory")
    parser.add_argument("--out_dir", required=True,
                        help="Output directory for results and per-fold CSVs")
    parser.add_argument("--src_cpu", type=str, default=None,
                        help="Source CPU ID filter (e.g. '0'). "
                             "Omit to load all CPUs (needed for arm_server 1000M).")
    parser.add_argument("--tgt_cpu", type=str, default=None,
                        help="Target CPU ID filter (e.g. '4' for edge OoO). "
                             "Omit to load all CPUs.")
    parser.add_argument("--suite", choices=["all", "spec_2017", "spec_2026", "dacapo_c2", "dacapo_c1"], default="spec_2017",
                        help="Benchmark suite to include: "
                             "'all', 'spec_2017' (default), 'spec_2026', 'dacapo_c2', or 'dacapo_c1'.")
    # --target_key is contributed by add_feature_args() below, shared with the
    # cross-frequency and cross-processor trainers so every experiment names the
    # prediction target the same way.
    parser.add_argument("--freq", type=str, default=None,
                        help="Run only the matched frequency pair at this value "
                             "(e.g. '1.0' to run only 1.0 GHz -> 1.0 GHz). "
                             "Omit to run all src × tgt frequency combinations.")
    parser.add_argument("--src_label", type=str, default=None,
                        help="Short name for source platform used in output paths "
                             "(default: basename of --src_data_dir).")
    parser.add_argument("--tgt_label", type=str, default=None,
                        help="Short name for target platform used in output paths "
                             "(default: basename of --tgt_data_dir).")
    add_feature_args(parser)
    args = parser.parse_args()

    src_label = args.src_label or os.path.basename(args.src_data_dir.rstrip("/"))
    tgt_label = args.tgt_label or os.path.basename(args.tgt_data_dir.rstrip("/"))

    # Append cpu ID to label when filtering a specific core
    if args.src_cpu is not None:
        src_label = f"{src_label}_cpu{args.src_cpu}"
    if args.tgt_cpu is not None:
        tgt_label = f"{tgt_label}_cpu{args.tgt_cpu}"

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  cross_sys_arm")
    print(f"  src : {src_label}  ({os.path.abspath(args.src_data_dir)})")
    print(f"  tgt : {tgt_label}  ({os.path.abspath(args.tgt_data_dir)})")
    print(f"  suite          : {args.suite}  freq={args.freq or 'all'}")
    print(f"  Features       | raw counters")
    print(f"  LOOCV          | strict={args.strict_loocv}  jobs={args.jobs}")
    print(f"{'='*60}\n")

    src_map = load_platform_data(args.src_data_dir, cpu_id=args.src_cpu, suite=args.suite)
    tgt_map = load_platform_data(args.tgt_data_dir, cpu_id=args.tgt_cpu, suite=args.suite)
    src_map = filter_excluded_benchmarks(src_map, args)
    tgt_map = filter_excluded_benchmarks(tgt_map, args)

    if not src_map:
        print(f"[ERROR] No data loaded from src: {args.src_data_dir}")
        return
    if not tgt_map:
        print(f"[ERROR] No data loaded from tgt: {args.tgt_data_dir}")
        return

    src_freqs = sorted({k[0] for k in src_map})
    tgt_freqs = sorted({k[0] for k in tgt_map})
    print(f"  {src_label} frequencies : {src_freqs}")
    print(f"  {tgt_label} frequencies : {tgt_freqs}")

    if args.freq is not None:
        src_freqs = [f for f in src_freqs if f == args.freq]
        tgt_freqs = [f for f in tgt_freqs if f == args.freq]
        if not src_freqs or not tgt_freqs:
            print(f"[ERROR] --freq {args.freq} not found in src or tgt data.")
            return
        print(f"  Filtering to matched frequency: {args.freq} GHz")

    all_summary = []
    for sf in src_freqs:
        for tf in tgt_freqs:
            print(f"\n  --- {src_label} {sf} GHz  ->  {tgt_label} {tf} GHz ---")
            try:
                res = run_freq_pair(
                    src_map, tgt_map, sf, tf,
                    src_label, tgt_label, args, args.out_dir,
                )
            except Exception as e:
                print(f"  [ERROR] Pair {src_label}@{sf}->{tgt_label}@{tf} failed: {e}")
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

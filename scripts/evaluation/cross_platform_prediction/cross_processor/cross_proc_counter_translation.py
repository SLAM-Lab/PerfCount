"""
cross_proc_counter_translation.py
==================================
Per-counter cross-processor translation for x86 heterogeneous platforms.

Structurally identical to cross_proc_x86.py, but the prediction target is the
first counter in --input_counters rather than ref_cycles.  Used to translate
each important counter from the source CPU's measurement space into what it
would look like on the target CPU — enabling the cross_proc_translated
heterogeneous-history variant in the workload forecasting sweep.

Typical usage
-------------
# Predict cpu_cycles (cpu16 -> cpu0), using top-4 counters as features
python cross_proc_counter_translation.py \\
    --data_dir ../../../../processed_data_10M/x86_desktop_heterogeneous \\
    --out_dir  ../../../../results/cross_platform/cross_proc/x86_10M/counter_translation/cpu16_to_cpu0/spec2017/cpu_cycles/top4 \\
    --src_cpu 16 --tgt_cpu 0 --suite spec2017 --strict_loocv \\
    --input_counters cpu_cycles branch_load_misses branch_misses dtlb_loads
"""

import os
import sys
import argparse

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
from cross_processor.cross_proc_x86 import load_cpu_data

import re as _re


def _suite_prefix(bench):
    """Map a benchmark name (e.g. dacapo_avrora, spec_505.mcf_r, spec_729.abc_r)
    to a suite label matching the --suite choices."""
    if bench.startswith("dacapo"):
        return "dacapo"
    m = _re.match(r"spec_(\d+)", bench)
    if m:
        return "spec2026" if int(m.group(1)) >= 700 else "spec2017"
    return "unknown"


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
        test_df    = restrict_input_counters(test_df,    "_src", getattr(args, "input_counters", None))

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

        pred_log  = model.predict(X_test)
        pred_vals = np.exp(pred_log) * test_df["source_val"].values

        y_true   = test_df["target_y"].values
        src_vals = test_df["source_val"].values

        m_ml   = compute_metrics(y_true, pred_vals)
        m_copy = compute_metrics(y_true, src_vals)

        os.makedirs(out_dir, exist_ok=True)
        model.save_model(os.path.join(out_dir, f"model_{test_bench}.cbm"))
        if getattr(args, "save_predictions", False):
            pd.DataFrame({
                "source_val":       src_vals,
                "target_actual":    y_true,
                "target_predicted": pred_vals,
                "ratio_actual":     y_true / src_clean_t.values,
                "ratio_predicted":  np.exp(pred_log),
            }).to_csv(os.path.join(out_dir, f"predictions_{test_bench}.csv"), index=False)

        return {
            "bench":               test_bench,
            "wmape":               m_ml["wmape"],
            "mape":                m_ml["mape"],
            "mdape":               m_ml["mdape"],
            "wmape_copy":          m_copy["wmape"],
            "mape_copy":           m_copy["mape"],
            "feature_importances": importances,
        }

    except Exception as e:
        print(f"  [WARN] Fold '{test_bench}' failed: {e}")
        return None


# =============================================================================
# 2.  FREQUENCY-PAIR RUNNER
# =============================================================================

def run_freq_pair(src_map, tgt_map, src_freq, tgt_freq, src_cpu, tgt_cpu, args, out_dir):
    target_counter = args.input_counters[0]

    benches_src = {k[1] for k in src_map if k[0] == src_freq}
    benches_tgt = {k[1] for k in tgt_map if k[0] == tgt_freq}
    common = sorted(benches_src & benches_tgt)

    if args.suite != "all":
        common = [b for b in common if _suite_prefix(b) == args.suite]

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
            target_key=target_counter,
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

    label = f"cpu{src_cpu} {src_freq} GHz -> cpu{tgt_cpu} {tgt_freq} GHz  [{target_counter}]"
    df_res = print_summary(results, label=label)
    df_res.to_csv(os.path.join(direction_dir, "per_fold_results.csv"), index=False)
    save_feature_importance(results, direction_dir)

    return {
        "pair":       f"cpu{src_cpu}@{src_freq}->cpu{tgt_cpu}@{tgt_freq}",
        "wmape_ml":   df_res["wmape"].mean(),
        "mape_ml":    df_res["mape"].mean(),
        "wmape_copy": df_res["wmape_copy"].mean(),
        "mape_copy":  df_res["mape_copy"].mean(),
        "n_folds":    len(df_res),
    }


# =============================================================================
# 3.  MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Per-counter cross-processor translation for x86 heterogeneous platforms. "
                    "The first counter in --input_counters is the prediction target."
    )
    parser.add_argument("--data_dir", required=True,
                        help="Directory containing aligned x86 CSV files")
    parser.add_argument("--out_dir", required=True,
                        help="Output directory for results and diagnostics")
    parser.add_argument("--src_cpu", required=True, type=str,
                        help="Source CPU ID (e.g. '16' for E-Core)")
    parser.add_argument("--tgt_cpu", required=True, type=str,
                        help="Target CPU ID (e.g. '0' for P-Core)")
    parser.add_argument("--suite", choices=["all", "dacapo", "spec2017", "spec2026"], default="all",
                        help="Benchmark suite to include (default: all).")
    parser.add_argument("--save_predictions", action="store_true", default=False,
                        help="Save per-fold prediction CSVs.")
    add_feature_args(parser)
    parser.add_argument("--force", action="store_true", default=False,
                        help="Re-run even if grand_summary.csv already exists.")
    args = parser.parse_args()

    if not args.input_counters:
        parser.error("--input_counters is required; the first counter is the prediction target.")

    target_counter = args.input_counters[0]

    os.makedirs(args.out_dir, exist_ok=True)
    summary_path = os.path.join(args.out_dir, "grand_summary.csv")
    if os.path.exists(summary_path) and not args.force:
        print(f"[SKIP] {summary_path} already exists. Use --force to re-run.")
        return

    print(f"\n{'='*60}")
    print(f"  cross_proc_counter_translation | data: {os.path.abspath(args.data_dir)}")
    print(f"  cpu{args.src_cpu} -> cpu{args.tgt_cpu}  |  target: {target_counter}")
    print(f"  input_counters: {args.input_counters}")
    print(f"  LOOCV: strict={args.strict_loocv}  jobs={args.jobs}  suite={args.suite}")
    print(f"{'='*60}\n")

    src_map = load_cpu_data(args.data_dir, args.src_cpu)
    tgt_map = load_cpu_data(args.data_dir, args.tgt_cpu)

    if not src_map:
        print(f"[ERROR] No data found for src cpu{args.src_cpu}. Check --data_dir.")
        return
    if not tgt_map:
        print(f"[ERROR] No data found for tgt cpu{args.tgt_cpu}. Check --data_dir.")
        return

    src_freqs = sorted({k[0] for k in src_map})
    tgt_freqs = sorted({k[0] for k in tgt_map})
    print(f"  cpu{args.src_cpu} frequencies: {src_freqs}")
    print(f"  cpu{args.tgt_cpu} frequencies: {tgt_freqs}")

    all_summary = []
    for sf in src_freqs:
        for tf in tgt_freqs:
            print(f"\n  --- cpu{args.src_cpu} {sf} GHz  ->  cpu{args.tgt_cpu} {tf} GHz  [{target_counter}] ---")
            try:
                res = run_freq_pair(
                    src_map, tgt_map, sf, tf,
                    args.src_cpu, args.tgt_cpu, args, args.out_dir,
                )
            except Exception as e:
                print(f"  [ERROR] Pair failed: {e}")
                res = None
            if res:
                all_summary.append(res)

    if all_summary:
        df_summary = pd.DataFrame(all_summary)
        df_summary.to_csv(summary_path, index=False)
        print(f"\n{'='*60}")
        print("  GRAND SUMMARY")
        print("=" * 60)
        print(df_summary.to_string(index=False))
        print(f"\n  Saved to: {summary_path}")


if __name__ == "__main__":
    main()

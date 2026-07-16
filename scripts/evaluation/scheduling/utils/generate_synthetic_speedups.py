"""
generate_synthetic_speedups.py
===============================
Generates predicted speedup CSVs for all configuration pairs on a given platform,
using trained CatBoost cross-config models instead of measured oracle data.

The output files have the same format as generate_speedup_matrix.py, allowing
the scheduling simulator to use them directly. Measured data (if available) is
included alongside predictions for comparison.

Model discovery convention
--------------------------
Cross-frequency (same CPU):
    {cf_model_root}/cpu{cpu_id}/{src_freq}GHz_to_{tgt_freq}GHz/model_all.cbm

Cross-processor / cross-processor+frequency (different CPU):
    {cp_model_root}/cpu{src_cpu}_{src_freq}GHz_to_cpu{tgt_cpu}_{tgt_freq}GHz/model_all.cbm

Usage
-----
# x86 — full counter set
python generate_synthetic_speedups.py \\
    --platform x86 \\
    --data_dir processed_data_10M/x86_desktop_heterogeneous \\
    --cf_model_root results/cross_platform/cross_freq/x86_10M \\
    --cp_model_root results/cross_platform/cross_proc/x86_10M \\
    --output_dir results/scheduling/synthetic_speedups/x86_10M/full \\
    --suite spec

# x86 — top-4 limited counter set
python generate_synthetic_speedups.py \\
    --platform x86 \\
    --data_dir processed_data_10M/x86_desktop_heterogeneous \\
    --cf_model_root results/cross_platform/cross_freq/x86_10M_top4 \\
    --cp_model_root results/cross_platform/cross_proc/x86_10M_top4 \\
    --output_dir results/scheduling/synthetic_speedups/x86_10M/top4 \\
    --suite spec

# arm_edge
python generate_synthetic_speedups.py \\
    --platform arm_edge \\
    --data_dir processed_data_10M/arm_edge_heterogeneous \\
    --cf_model_root results/cross_platform/cross_freq/arm_edge_10M \\
    --cp_model_root results/cross_platform/cross_proc/arm_edge_10M \\
    --output_dir results/scheduling/synthetic_speedups/arm_edge_10M/full \\
    --suite spec
"""

import os
import sys
import argparse
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

# Add cross_platform_prediction to path for shared_features
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_EVAL_DIR   = os.path.abspath(os.path.join(_SCRIPT_DIR, "../../"))
_XPLAT_DIR  = os.path.join(_EVAL_DIR, "cross_platform_prediction")
sys.path.insert(0, _XPLAT_DIR)

from shared_features import add_feature_args, build_features

try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None


# =============================================================================
# PLATFORM CONFIGURATION
# =============================================================================

PLATFORM_CONFIGS = {
    "x86": {
        # cpu_id → (core_type_label, [available_freqs])
        "cpus": {
            "0":  ("P",  ["1.0", "2.0", "3.0", "4.0"]),
            "16": ("E",  ["1.0", "2.0", "3.0", "4.0"]),
        },
    },
    "arm_edge": {
        "cpus": {
            "4": ("OOO", ["1.0", "2.0"]),
            "1": ("InO", ["1.0"]),
        },
    },
}


# =============================================================================
# MODEL MANAGEMENT
# =============================================================================

_model_cache = {}


def load_model(model_path):
    """Load and cache a CatBoost model from disk."""
    if model_path in _model_cache:
        return _model_cache[model_path]
    if CatBoostRegressor is None:
        raise ImportError("catboost is not installed")
    if not os.path.exists(model_path):
        return None
    model = CatBoostRegressor()
    model.load_model(model_path)
    _model_cache[model_path] = model
    return model


def find_model(src_cpu, src_freq, tgt_cpu, tgt_freq, cf_model_root, cp_model_root):
    """
    Return path to the appropriate model for src→tgt, or None if not found.

    Same CPU → cross-frequency model.
    Different CPU → cross-processor model (covers same-freq and cross-freq).
    """
    if src_cpu == tgt_cpu:
        path = os.path.join(
            cf_model_root, f"cpu{src_cpu}",
            f"{src_freq}GHz_to_{tgt_freq}GHz", "model_all.cbm"
        )
    else:
        path = os.path.join(
            cp_model_root,
            f"cpu{src_cpu}_{src_freq}GHz_to_cpu{tgt_cpu}_{tgt_freq}GHz",
            "model_all.cbm"
        )
    return path if os.path.exists(path) else None


# =============================================================================
# FEATURE BUILDING
# =============================================================================

def _default_feature_args():
    """Return an argparse.Namespace with all feature flags at their defaults."""
    p = argparse.ArgumentParser()
    add_feature_args(p)
    return p.parse_args([])


def build_feature_matrix(df_src, feature_args):
    """
    Build the feature matrix from a source aligned CSV DataFrame.
    Adds the '_src' suffix expected by build_features().
    """
    df = df_src.copy()
    df.columns = [f"{c}_src" for c in df.columns]
    return build_features(df, suffix="_src", args=feature_args)


# =============================================================================
# PREDICTION
# =============================================================================

def predict_speedup(model, df_src, src_freq, tgt_freq, feature_args):
    """
    Apply a cross-config model to source counter data.

    Returns an array of predicted speedups (src_time / tgt_time).
    """
    X = build_feature_matrix(df_src, feature_args)
    if X.empty:
        return None

    feat_names = model.feature_names_
    X_aligned  = X.reindex(columns=feat_names, fill_value=0)

    # CatBoost categorical columns must be str
    try:
        cat_indices = model.get_cat_feature_indices()
        for ci in cat_indices:
            col = X_aligned.columns[ci]
            X_aligned[col] = X_aligned[col].astype(str)
    except Exception:
        pass

    pred_log   = model.predict(X_aligned)
    pred_ratio = np.exp(pred_log)   # = tgt_ref_cycles / src_ref_cycles = tgt_time / src_time

    # No frequency correction: ref_cycles already encodes wall-clock time at a fixed
    # reference frequency, so pred_ratio directly gives the time ratio.
    # speedup = src_time / tgt_time  (> 1 means target is faster)
    speedup = 1.0 / np.clip(pred_ratio, 1e-6, 1e6)
    return speedup


# =============================================================================
# WORKLOAD FORECASTING PLACEHOLDER
# =============================================================================

def predict_future_speedup_wf(df_src_history, wf_model_dir, horizon,
                               cross_config_model, src_freq, tgt_freq):
    """
    PLACEHOLDER: Workload forecasting integration for proactive scheduling.

    Given the recent history of source-config counter observations and a
    per-workload forecasting model, predict counter values at future timesteps
    (horizon chunks ahead), then apply the cross-config model to those predicted
    counters to estimate future speedup.

    Parameters
    ----------
    df_src_history : DataFrame   — recent source-config counter observations
    wf_model_dir   : str         — directory of saved per-workload forecasting models
    horizon        : int         — how many chunks ahead to predict
    cross_config_model           — loaded CatBoostRegressor for src→tgt
    src_freq, tgt_freq : str     — GHz strings for frequency correction

    Returns
    -------
    None  (not yet implemented — returns None to signal no prediction available)
    """
    # TODO:
    # 1. Load the per-workload forecasting model from wf_model_dir
    # 2. Use df_src_history to predict counter values at t+1 .. t+horizon
    # 3. For each predicted future counter vector, apply cross_config_model
    #    to get predicted future speedup
    # 4. Return array of shape (horizon,) with predicted future speedups
    return None


# =============================================================================
# SPEEDUP CSV GENERATION
# =============================================================================

def generate_for_workload_phase(
    workload, phase, src_cpu, src_freq, src_label,
    data_dir, platform_cfg, cf_model_root, cp_model_root,
    output_dir, suite_filter, feature_args
):
    """
    Generate a predicted speedup CSV for one (src_config, workload, phase).
    """
    src_pattern = os.path.join(
        data_dir,
        f"aligned_{workload}_{src_freq}GHz_cpu{src_cpu}_phase{phase}.csv"
    )
    matches = glob.glob(src_pattern)
    if not matches:
        return

    df_src = pd.read_csv(matches[0])
    df_src.columns = [c.strip() for c in df_src.columns]
    if "instructions" in df_src.columns and "cpu_cycles" in df_src.columns:
        df_src = df_src[(df_src["instructions"] > 100_000) & (df_src["cpu_cycles"] > 0)]
    if df_src.empty:
        return

    # Measured source time (ref_cycles converted to seconds at 1 GHz reference)
    if "ref_cycles" not in df_src.columns:
        return
    src_time = df_src["ref_cycles"].values / 1e9

    out_rows = {"sample_index": df_src["sample_index"].values if "sample_index" in df_src.columns
                else np.arange(len(df_src))}
    out_rows[f"Time_{src_label}"] = src_time

    for tgt_cpu, (tgt_label_base, tgt_freqs) in platform_cfg["cpus"].items():
        for tgt_freq in tgt_freqs:
            tgt_label = f"{tgt_label_base}_{tgt_freq}GHz"
            col_name  = f"Predicted_Speedup_{tgt_label}_vs_{src_label}"

            if tgt_cpu == src_cpu and tgt_freq == src_freq:
                continue  # skip self

            model_path = find_model(src_cpu, src_freq, tgt_cpu, tgt_freq,
                                    cf_model_root, cp_model_root)
            if model_path is None:
                continue

            model = load_model(model_path)
            if model is None:
                continue

            speedup = predict_speedup(model, df_src, src_freq, tgt_freq, feature_args)
            if speedup is None or len(speedup) != len(df_src):
                continue

            out_rows[col_name] = speedup

            # Placeholder hook: future workload forecasting would add
            # Forecast_Speedup_{tgt_label}_vs_{src_label} columns here
            # using predict_future_speedup_wf()

    if len(out_rows) <= 2:  # only sample_index + Time_*, no predictions
        print(f"    [WARN] No predictions generated for {src_label} {workload} phase{phase}")
        return

    out_df = pd.DataFrame(out_rows)
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"speedups_{src_label}_{workload}_phase{phase}.csv")
    out_df.to_csv(out_file, index=False)
    print(f"  → {os.path.basename(out_file)}  ({len(out_df)} chunks, "
          f"{len(out_df.columns)-2} speedup columns)")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate predicted speedup CSVs using trained cross-config CatBoost models."
    )
    parser.add_argument("--platform",      required=True,
                        choices=list(PLATFORM_CONFIGS.keys()),
                        help="Hardware platform")
    parser.add_argument("--data_dir",      required=True,
                        help="Directory of aligned source CSVs")
    parser.add_argument("--cf_model_root", required=True,
                        help="Root of cross-frequency models. "
                             "Models at {root}/cpu{id}/{src}GHz_to_{tgt}GHz/model_all.cbm")
    parser.add_argument("--cp_model_root", required=True,
                        help="Root of cross-processor models. "
                             "Models at {root}/cpu{src}_{sf}GHz_to_cpu{tgt}_{tf}GHz/model_all.cbm")
    parser.add_argument("--output_dir",    required=True,
                        help="Output directory for predicted speedup CSVs")
    parser.add_argument("--suite",         default="all",
                        choices=["all", "spec", "dacapo"],
                        help="Benchmark suite filter")
    add_feature_args(parser)
    args = parser.parse_args()

    platform_cfg = PLATFORM_CONFIGS[args.platform]
    feature_args = _default_feature_args()
    # Propagate any feature toggles explicitly passed on the command line
    for attr in ("use_mpki", "use_miss_rates", "use_stall_rates",
                 "use_bottleneck_class", "rolling_window", "use_raw_counters",
                 "input_counters", "exclude_features"):
        if hasattr(args, attr):
            setattr(feature_args, attr, getattr(args, attr))

    data_dir     = os.path.abspath(args.data_dir)
    output_dir   = os.path.abspath(args.output_dir)
    cf_root      = os.path.abspath(args.cf_model_root)
    cp_root      = os.path.abspath(args.cp_model_root)

    # Discover all aligned CSVs
    pattern = re.compile(
        r"aligned_(?P<bench>.+?)_(?P<freq>[\d.]+)GHz_cpu(?P<cpu>\d+)_phase(?P<phase>\d+)\.csv"
    )
    found = []
    for fpath in glob.glob(os.path.join(data_dir, "aligned_*.csv")):
        m = pattern.match(os.path.basename(fpath))
        if not m:
            continue
        bench = m.group("bench")
        if args.suite != "all":
            if args.suite == "spec" and not bench.startswith("spec_"):
                continue
            if args.suite == "dacapo" and not bench.startswith("dacapo_"):
                continue
        cpu   = m.group("cpu")
        freq  = m.group("freq")
        phase = m.group("phase")
        if cpu in platform_cfg["cpus"] and freq in platform_cfg["cpus"][cpu][1]:
            found.append((bench, phase, cpu, freq))

    # Deduplicate (multiple files can share the same key)
    found = sorted(set(found))
    print(f"Found {len(found)} source (workload, phase, cpu, freq) combinations.")

    for bench, phase, src_cpu, src_freq in found:
        src_label = f"{platform_cfg['cpus'][src_cpu][0]}_{src_freq}GHz"
        print(f"  {src_label}  {bench}  phase{phase}")
        generate_for_workload_phase(
            workload=bench, phase=phase,
            src_cpu=src_cpu, src_freq=src_freq, src_label=src_label,
            data_dir=data_dir, platform_cfg=platform_cfg,
            cf_model_root=cf_root, cp_model_root=cp_root,
            output_dir=output_dir, suite_filter=args.suite,
            feature_args=feature_args,
        )

    print(f"\nDone. Predicted speedup CSVs written to: {output_dir}")


if __name__ == "__main__":
    main()

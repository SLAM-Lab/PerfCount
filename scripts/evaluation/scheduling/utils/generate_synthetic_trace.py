"""
generate_synthetic_trace.py
===========================
Apply a trained CatBoost cross-frequency model to an existing speedup CSV and
annotate it with model-predicted columns alongside the measured ones.

Usage
-----
python generate_synthetic_trace.py \
    --speedup   results/scheduling/granular_phase_traces_10M/speedups_dacapo_avrora_phase0.csv \
    --aligned   processed_data/x86_desktop_heterogeneous/aligned_dacapo_avrora_4.0GHz_cpu0_phase0.csv \
    --model     results/cf_x86/4.0GHz_to_2.2GHz/model_all.cbm \
    --src-config P_4.0GHz \
    --tgt-config E_2.2GHz \
    --output    results/scheduling/synthetic/speedups_dacapo_avrora_phase0_synth.csv

Multiple model/config triplets are supported by repeating the flags:
    --model m1.cbm --src-config P_4.0GHz --tgt-config E_2.2GHz \
    --model m2.cbm --src-config P_4.0GHz --tgt-config E_3.0GHz

The speedup CSV is read, the new columns are appended, and the result is written
to --output (defaults to overwriting --speedup if --output is omitted).
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

# Resolve path to shared_features.py (two levels up from this file's directory)
_UTILS_DIR  = os.path.dirname(os.path.abspath(__file__))
_SCHED_DIR  = os.path.dirname(_UTILS_DIR)
_EVAL_DIR   = os.path.dirname(_SCHED_DIR)
_XPLAT_DIR  = os.path.join(_EVAL_DIR, "cross_platform_prediction")
sys.path.insert(0, _XPLAT_DIR)

from shared_features import add_feature_args, build_features, cat_feature_names  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_freq(config_str: str) -> float:
    """Extract GHz value from config name like 'P_4.0GHz' or 'E_2.2GHz'."""
    parts = config_str.split("_")
    for p in reversed(parts):
        if p.endswith("GHz"):
            try:
                return float(p[:-3])
            except ValueError:
                pass
    raise ValueError(f"Cannot extract frequency from config name '{config_str}'")


def _default_feature_args():
    """Return a Namespace with all feature toggles at their defaults (all on)."""
    parser = argparse.ArgumentParser()
    add_feature_args(parser)
    return parser.parse_args([])


def apply_model(aligned_df: pd.DataFrame, model: CatBoostRegressor,
                src_config: str, tgt_config: str,
                speedup_df: pd.DataFrame, args) -> dict:
    """
    Run one model against the aligned trace and return a dict of new columns.

    Parameters
    ----------
    aligned_df  : DataFrame of raw perf counters from the source aligned CSV
    model       : Loaded CatBoostRegressor
    src_config  : e.g. 'P_4.0GHz'
    tgt_config  : e.g. 'E_2.2GHz'
    speedup_df  : The existing speedup CSV (used for Time_{src_config} column)
    args        : feature-toggle Namespace

    Returns
    -------
    dict mapping new column names to Series/arrays aligned to speedup_df rows
    """
    src_freq = _parse_freq(src_config)
    tgt_freq = _parse_freq(tgt_config)

    # Build features from aligned CSV (add _src suffix as build_features expects)
    df_src = aligned_df.add_suffix("_src")
    X_all = build_features(df_src, suffix="_src", args=args)

    if X_all.empty:
        print(f"  [WARN] Feature matrix is empty for {src_config}->{tgt_config}, skipping.")
        return {}

    # Align to model's expected features
    feat_names = model.feature_names_
    X = X_all.reindex(columns=feat_names, fill_value=0)

    # CatBoost categoricals must be str
    cat_indices = model.get_cat_feature_indices()
    for idx in cat_indices:
        col = X.columns[idx]
        X[col] = X[col].astype(str)

    # Truncate/pad to match speedup_df length
    n_speedup = len(speedup_df)
    n_feat    = len(X)
    if n_feat != n_speedup:
        print(f"  [WARN] Aligned CSV has {n_feat} rows but speedup CSV has {n_speedup} rows; "
              f"truncating to min({n_feat}, {n_speedup}).")
    n = min(n_feat, n_speedup)
    X = X.iloc[:n].reset_index(drop=True)

    pred_log   = model.predict(X)
    pred_ratio = np.exp(pred_log)   # = tgt_cpu_cycles / src_cpu_cycles

    # time ratio = tgt_time / src_time = pred_ratio * (src_freq / tgt_freq)
    # speedup    = src_time / tgt_time = 1 / time_ratio
    time_ratio         = pred_ratio * (src_freq / tgt_freq)
    predicted_speedup  = 1.0 / time_ratio

    speedup_col = f"Predicted_Speedup_{tgt_config}_vs_{src_config}"
    time_col    = f"Predicted_Time_{tgt_config}"

    # Derive predicted absolute time from the measured source time if available
    src_time_col = f"Time_{src_config}"
    result = {speedup_col: np.full(n_speedup, np.nan),
              time_col:    np.full(n_speedup, np.nan)}

    result[speedup_col][:n] = predicted_speedup

    if src_time_col in speedup_df.columns:
        src_times = speedup_df[src_time_col].values[:n]
        result[time_col][:n] = src_times * time_ratio
    else:
        print(f"  [INFO] Column '{src_time_col}' not in speedup CSV; "
              f"'{time_col}' will be NaN.")

    # Report accuracy vs measured speedup if the measured column exists
    meas_col = f"Speedup_{tgt_config}_vs_{src_config}"
    if meas_col in speedup_df.columns:
        meas = speedup_df[meas_col].values[:n]
        valid = meas > 0
        if valid.any():
            mape = np.mean(np.abs(meas[valid] - predicted_speedup[valid]) / meas[valid]) * 100
            corr = np.corrcoef(meas[valid], predicted_speedup[valid])[0, 1]
            print(f"  {src_config}->{tgt_config}: MAPE={mape:.1f}%  corr={corr:.3f}  "
                  f"(vs measured '{meas_col}')")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Annotate a speedup CSV with cross-frequency model predictions."
    )
    parser.add_argument("--speedup", required=True,
                        help="Input speedup CSV (output of generate_speedup_matrix.py)")
    parser.add_argument("--aligned", required=True,
                        help="Aligned perf-counter CSV for the source configuration")
    parser.add_argument("--model", action="append", dest="models", metavar="PATH",
                        required=True,
                        help="Path to .cbm model file (repeat for multiple models)")
    parser.add_argument("--src-config", action="append", dest="src_configs", metavar="CFG",
                        required=True,
                        help="Source config name matching the aligned CSV (e.g. P_4.0GHz)")
    parser.add_argument("--tgt-config", action="append", dest="tgt_configs", metavar="CFG",
                        required=True,
                        help="Target config name (e.g. E_2.2GHz)")
    parser.add_argument("--output", default=None,
                        help="Output CSV path. Defaults to overwriting --speedup.")
    add_feature_args(parser)
    args = parser.parse_args()

    if not (len(args.models) == len(args.src_configs) == len(args.tgt_configs)):
        parser.error("--model, --src-config, and --tgt-config must appear the same number of times.")

    # Load inputs
    speedup_df = pd.read_csv(args.speedup)
    aligned_df = pd.read_csv(args.aligned)
    aligned_df.columns = [c.strip() for c in aligned_df.columns]
    print(f"Speedup CSV : {len(speedup_df)} rows  |  {args.speedup}")
    print(f"Aligned CSV : {len(aligned_df)} rows  |  {args.aligned}")
    print(f"Models      : {len(args.models)}")

    for model_path, src_cfg, tgt_cfg in zip(args.models, args.src_configs, args.tgt_configs):
        print(f"\n  Loading model: {model_path}")
        if not os.path.exists(model_path):
            print(f"  [ERROR] Model file not found: {model_path}")
            continue

        model = CatBoostRegressor()
        model.load_model(model_path)

        new_cols = apply_model(aligned_df, model, src_cfg, tgt_cfg, speedup_df, args)
        for col_name, values in new_cols.items():
            speedup_df[col_name] = values

    out_path = args.output or args.speedup
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    speedup_df.to_csv(out_path, index=False)
    print(f"\nWrote {len(speedup_df)} rows, {len(speedup_df.columns)} columns → {out_path}")


if __name__ == "__main__":
    main()

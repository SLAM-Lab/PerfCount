"""
cross_freq_precompute.py
========================
Offline inference pass for cross-frequency CatBoost models.

For each SPEC2017 workload + phase + P-core source frequency, loads the aligned
PMU CSV, runs build_features(), and calls each of the three cross-freq models to
produce model-predicted speedup ratios to the other 3 P-core frequencies.

Output: one CSV per (workload, phase, src_freq) in
    results/scheduling/model_predictions/speedups_from_P_<ghz>GHz/
        speedups_P_<ghz>GHz_<bench>_phase<ph>.csv

CSV columns (matches data_loader.load_phase_data format):
    sample_index, Time_P_<src>GHz,
    Speedup_P_<tgt1>GHz_vs_P_<src>GHz,
    Speedup_P_<tgt2>GHz_vs_P_<src>GHz,
    Speedup_P_<tgt3>GHz_vs_P_<src>GHz

MAPE vs oracle speedup columns is printed per workload for validation.

Usage:
    python cross_freq_precompute.py \
        --model_dir results/cross_platform/cross_freq/x86_10M/cpu0/spec2017/full \
        --pmu_dir processed_data_10M/x86_desktop_heterogeneous \
        --oracle_dir results/scheduling/speedup_test/granular_phase_traces \
        --out_dir results/scheduling/model_predictions
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "cross_platform_prediction"))
from shared_features import build_features, try_load_model

P_FREQS = [1.0, 2.0, 3.0, 4.0]


def make_feature_args():
    import argparse
    ns = argparse.Namespace(
        use_mpki=True,
        use_miss_rates=True,
        use_stall_rates=True,
        use_bottleneck_class=True,
        exclude_features=[],
        rolling_window=0,
        input_counters=None,
    )
    return ns


def find_workloads(model_dir):
    """Discover bench names from model CBM files in the reference freq-pair dir."""
    ref_dir = Path(model_dir) / "1.0GHz_to_2.0GHz"
    benches = sorted(
        p.stem.replace("model_", "")
        for p in ref_dir.glob("model_*.cbm")
    )
    return benches


def find_phases(pmu_dir, bench, src_freq):
    """Return sorted phase indices that have an aligned CSV for (bench, src_freq)."""
    pat = f"aligned_{bench}_{src_freq:.1f}GHz_cpu0_phase*.csv"
    files = sorted(Path(pmu_dir).glob(pat))
    phases = []
    for f in files:
        stem = f.stem  # aligned_<bench>_<ghz>GHz_cpu0_phase<ph>
        ph_str = stem.rsplit("phase", 1)[-1]
        phases.append(int(ph_str))
    return phases


def load_oracle_speedups(oracle_dir, bench, ph, src_freq):
    """Return oracle speedup dict {tgt_cfg: speedup_series} from the oracle CSV."""
    fname = f"speedups_P_{src_freq:.1f}GHz_{bench}_phase{ph}.csv"
    fpath = Path(oracle_dir) / fname
    if not fpath.exists():
        return None, None
    df = pd.read_csv(fpath)
    oracle_time = df[f"Time_P_{src_freq:.1f}GHz"].values
    speedups = {}
    for col in df.columns:
        if col.startswith("Speedup_P_") and "_vs_P_" in col:
            tgt = col.split("_vs_")[0].replace("Speedup_", "")
            speedups[tgt] = df[col].values
    return oracle_time, speedups


def run_precompute(model_dir, pmu_dir, oracle_dir, out_dir):
    args = make_feature_args()
    model_dir = Path(model_dir)
    pmu_dir = Path(pmu_dir)
    oracle_dir = Path(oracle_dir)
    out_dir = Path(out_dir)

    benches = find_workloads(model_dir)
    print(f"Found {len(benches)} workloads: {benches[:3]}...")

    all_mape = []

    for bench in benches:
        bench_mapes = []
        for src_freq in P_FREQS:
            tgt_freqs = [f for f in P_FREQS if f != src_freq]
            src_ghz = f"{src_freq:.1f}GHz"
            out_subdir = out_dir / f"speedups_from_P_{src_ghz}"
            out_subdir.mkdir(parents=True, exist_ok=True)

            # Load models for this src freq (one per target freq)
            models = {}
            for tgt_freq in tgt_freqs:
                tgt_ghz = f"{tgt_freq:.1f}GHz"
                pair_dir = model_dir / f"{src_ghz}_to_{tgt_ghz}"
                model = try_load_model(str(pair_dir), bench)
                if model is None:
                    print(f"  [WARN] No model for {bench} {src_ghz}->{tgt_ghz}, skipping")
                else:
                    models[tgt_freq] = model

            if not models:
                continue

            phases = find_phases(pmu_dir, bench, src_freq)
            if not phases:
                print(f"  [WARN] No aligned CSVs for {bench} @ {src_ghz}")
                continue

            for ph in phases:
                pmu_file = pmu_dir / f"aligned_{bench}_{src_ghz}_cpu0_phase{ph}.csv"
                try:
                    df_pmu = pd.read_csv(pmu_file)
                except Exception as e:
                    print(f"  [WARN] Could not read {pmu_file}: {e}")
                    continue

                X = build_features(df_pmu, "", args)
                if X.empty:
                    print(f"  [WARN] Empty features for {bench} {src_ghz} phase{ph}")
                    continue

                n = len(X)

                # Oracle data for this source freq + phase
                oracle_time, oracle_speedups = load_oracle_speedups(
                    oracle_dir, bench, ph, src_freq
                )
                if oracle_time is None:
                    print(f"  [WARN] No oracle CSV for {bench} {src_ghz} phase{ph}")
                    continue

                # Truncate to shortest length (model rows may differ from oracle)
                n_out = min(n, len(oracle_time))

                out_rows = {"sample_index": np.arange(n_out)}
                out_rows[f"Time_P_{src_ghz}"] = oracle_time[:n_out]

                for tgt_freq, model in models.items():
                    tgt_ghz = f"{tgt_freq:.1f}GHz"
                    col = f"Speedup_P_{tgt_ghz}_vs_P_{src_ghz}"

                    # Model predicts log(ref_cycles_tgt / ref_cycles_src) = log(time_tgt/time_src)
                    # Speedup = time_src/time_tgt = 1 / exp(model.predict(X))
                    pred_time_ratio = np.exp(model.predict(X.iloc[:n_out]))
                    pred_speedup = 1.0 / pred_time_ratio
                    out_rows[col] = pred_speedup

                    # MAPE vs oracle
                    oracle_key = f"P_{tgt_ghz}"
                    if oracle_speedups and oracle_key in oracle_speedups:
                        y_true = oracle_speedups[oracle_key][:n_out]
                        mask = y_true > 0
                        if mask.sum() > 0:
                            mape = np.mean(np.abs(y_true[mask] - pred_speedup[mask])
                                           / (y_true[mask] + 1e-9)) * 100
                            bench_mapes.append(mape)

                out_csv = out_subdir / f"speedups_P_{src_ghz}_{bench}_phase{ph}.csv"
                pd.DataFrame(out_rows).to_csv(out_csv, index=False)

        if bench_mapes:
            mean_mape = np.mean(bench_mapes)
            print(f"  {bench}: MAPE={mean_mape:.1f}% over {len(bench_mapes)} (wl,phase,pair) combos")
            all_mape.extend(bench_mapes)

    if all_mape:
        print(f"\nOverall MAPE: {np.mean(all_mape):.2f}% (median {np.median(all_mape):.2f}%)")
    print("Done.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir",  required=True,
                        help="results/cross_platform/cross_freq/x86_10M/cpu0/spec2017/full")
    parser.add_argument("--pmu_dir",    required=True,
                        help="processed_data_10M/x86_desktop_heterogeneous")
    parser.add_argument("--oracle_dir", required=True,
                        help="results/scheduling/speedup_test/granular_phase_traces")
    parser.add_argument("--out_dir",    required=True,
                        help="results/scheduling/model_predictions")
    args = parser.parse_args()
    run_precompute(args.model_dir, args.pmu_dir, args.oracle_dir, args.out_dir)


if __name__ == "__main__":
    main()

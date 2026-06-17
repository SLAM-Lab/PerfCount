"""
cross_proc_precompute.py
========================
Offline inference pass for cross-processor CatBoost models (P<->E migrations).

For each SPEC2017 workload + phase:
  P-core sources (cpu0): loads aligned PMU CSVs and predicts to all 4 E-core
    frequencies via cpu0_to_cpu16 models.
  E-core sources (cpu16): loads aligned PMU CSVs and predicts to all 4 P-core
    frequencies via cpu16_to_cpu0 models.

Output: one CSV per (workload, phase, src_core, src_freq) in
    results/scheduling/cross_proc_predictions/speedups_from_{P|E}_<ghz>GHz/
        {bench}_phase<ph>.csv

CSV columns:
  P-source CSVs:  Time_P_{src}GHz, Speedup_E_{tgt}GHz_vs_P_{src}GHz (x4 targets)
  E-source CSVs:  Time_E_{src}GHz, Speedup_P_{tgt}GHz_vs_E_{src}GHz (x4 targets)

MAPE vs oracle cross-proc speedup columns is printed per workload for validation.

Usage:
    python cross_proc_precompute.py \\
        --model_dir results/cross_platform/cross_proc/x86_10M \\
        --pmu_dir processed_data_10M/x86_desktop_heterogeneous \\
        --oracle_dir results/scheduling/speedup_test/granular_phase_traces \\
        --out_dir results/scheduling/cross_proc_predictions
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "cross_platform_prediction"))
from shared_features import build_features, try_load_model

FREQS = [1.0, 2.0, 3.0, 4.0]


def make_feature_args():
    return argparse.Namespace(
        use_mpki=True,
        use_miss_rates=True,
        use_stall_rates=True,
        use_bottleneck_class=True,
        exclude_features=[],
        rolling_window=0,
        input_counters=None,
    )


def find_workloads(p_to_e_model_dir):
    """Discover bench names from P->E model CBM files in the reference pair dir."""
    ref_dir = Path(p_to_e_model_dir) / "cpu0_1.0GHz_to_cpu16_1.0GHz"
    return sorted(p.stem.replace("model_", "") for p in ref_dir.glob("model_*.cbm"))


def find_phases(pmu_dir, bench, freq, core_suffix):
    """Return sorted phase indices for (bench, freq, core_suffix='cpu0'|'cpu16')."""
    phases = []
    for f in sorted(Path(pmu_dir).glob(
            f"aligned_{bench}_{freq:.1f}GHz_{core_suffix}_phase*.csv")):
        phases.append(int(f.stem.rsplit("phase", 1)[-1]))
    return phases


def load_oracle_speedups(oracle_dir, bench, ph, src_core, src_freq, tgt_core):
    """Return (oracle_time_src, {tgt_cfg: speedup_series}) from oracle CSV."""
    fname = f"speedups_{src_core}_{src_freq:.1f}GHz_{bench}_phase{ph}.csv"
    time_col = f"Time_{src_core}_{src_freq:.1f}GHz"
    fpath = Path(oracle_dir) / fname
    if not fpath.exists():
        return None, None
    df = pd.read_csv(fpath)
    if time_col not in df.columns:
        return None, None
    oracle_time = df[time_col].values
    speedups = {}
    for col in df.columns:
        if col.startswith(f"Speedup_{tgt_core}_") and f"_vs_{src_core}_" in col:
            tgt_cfg = col.split("_vs_")[0].replace("Speedup_", "")
            speedups[tgt_cfg] = df[col].values
    return oracle_time, speedups


def _process_direction(bench, src_freqs, src_core, tgt_core,
                       model_subdir, core_suffix,
                       pmu_dir, oracle_dir, out_dir, feat_args):
    """Run precompute for one direction (P->E or E->P). Returns list of MAPE values."""
    mapes = []
    for src_freq in src_freqs:
        src_ghz = f"{src_freq:.1f}GHz"
        out_subdir = out_dir / f"speedups_from_{src_core}_{src_ghz}"
        out_subdir.mkdir(parents=True, exist_ok=True)

        models = {}
        for tgt_freq in FREQS:
            tgt_ghz = f"{tgt_freq:.1f}GHz"
            pair_dir = model_subdir / f"{core_suffix[0]}_{src_ghz}_to_{core_suffix[1]}_{tgt_ghz}"
            model = try_load_model(str(pair_dir), bench)
            if model is not None:
                models[tgt_freq] = model

        if not models:
            continue

        phases = find_phases(pmu_dir, bench, src_freq, core_suffix[0])
        for ph in phases:
            pmu_file = pmu_dir / f"aligned_{bench}_{src_ghz}_{core_suffix[0]}_phase{ph}.csv"
            try:
                df_pmu = pd.read_csv(pmu_file)
            except Exception as e:
                print(f"  [WARN] {pmu_file.name}: {e}")
                continue

            X = build_features(df_pmu, "", feat_args)
            if X.empty:
                continue

            oracle_time, oracle_speedups = load_oracle_speedups(
                oracle_dir, bench, ph, src_core, src_freq, tgt_core)
            if oracle_time is None:
                print(f"  [WARN] No oracle for {bench} {src_core}_{src_ghz} phase{ph}")
                continue

            n_out = min(len(X), len(oracle_time))
            out_rows = {f"Time_{src_core}_{src_ghz}": oracle_time[:n_out]}

            for tgt_freq, model in models.items():
                tgt_ghz = f"{tgt_freq:.1f}GHz"
                col = f"Speedup_{tgt_core}_{tgt_ghz}_vs_{src_core}_{src_ghz}"
                pred_speedup = 1.0 / np.exp(model.predict(X.iloc[:n_out]))
                out_rows[col] = pred_speedup

                oracle_key = f"{tgt_core}_{tgt_ghz}"
                if oracle_speedups and oracle_key in oracle_speedups:
                    y_true = oracle_speedups[oracle_key][:n_out]
                    mask = y_true > 0
                    if mask.sum() > 0:
                        mape = np.mean(
                            np.abs(y_true[mask] - pred_speedup[mask])
                            / (np.abs(y_true[mask]) + 1e-9)
                        ) * 100
                        mapes.append(mape)

            out_csv = out_subdir / f"{bench}_phase{ph}.csv"
            pd.DataFrame(out_rows).to_csv(out_csv, index=False)

    return mapes


def run_precompute(model_dir, pmu_dir, oracle_dir, out_dir):
    feat_args = make_feature_args()
    model_dir = Path(model_dir)
    pmu_dir = Path(pmu_dir)
    oracle_dir = Path(oracle_dir)
    out_dir = Path(out_dir)

    p_to_e_dir = model_dir / "cpu0_to_cpu16" / "spec2017" / "full"
    e_to_p_dir = model_dir / "cpu16_to_cpu0" / "spec2017" / "full"

    benches = find_workloads(p_to_e_dir)
    print(f"Found {len(benches)} workloads: {benches[:3]}...")

    all_mape = []
    for bench in benches:
        bench_mapes = []

        bench_mapes += _process_direction(
            bench, FREQS, 'P', 'E', p_to_e_dir, ('cpu0', 'cpu16'),
            pmu_dir, oracle_dir, out_dir, feat_args)

        bench_mapes += _process_direction(
            bench, FREQS, 'E', 'P', e_to_p_dir, ('cpu16', 'cpu0'),
            pmu_dir, oracle_dir, out_dir, feat_args)

        if bench_mapes:
            print(f"  {bench}: MAPE={np.mean(bench_mapes):.1f}% ({len(bench_mapes)} combos)")
            all_mape.extend(bench_mapes)

    if all_mape:
        print(f"\nOverall MAPE: {np.mean(all_mape):.2f}% (median {np.median(all_mape):.2f}%)")
    print("Done.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir", required=True,
                        help="results/cross_platform/cross_proc/x86_10M")
    parser.add_argument("--pmu_dir", required=True,
                        help="processed_data_10M/x86_desktop_heterogeneous")
    parser.add_argument("--oracle_dir", required=True,
                        help="results/scheduling/speedup_test/granular_phase_traces")
    parser.add_argument("--out_dir", required=True,
                        help="results/scheduling/cross_proc_predictions")
    args = parser.parse_args()
    run_precompute(args.model_dir, args.pmu_dir, args.oracle_dir, args.out_dir)


if __name__ == "__main__":
    main()

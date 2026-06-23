"""
cross_freq_precompute.py
========================
Offline inference pass for cross-frequency CatBoost models.

For each workload + phase + source frequency, loads the aligned PMU CSV,
runs build_features(), and calls each of the three cross-freq models to
produce model-predicted speedup ratios to the other 3 frequencies.

Supports multiple benchmark suites (spec2017, spec2026, dacapo) via the
--suites flag.  Models for each suite are expected under:
    <model_base_dir>/<cpu>/<suite>/full/

Output: one CSV per (workload, phase, src_freq) in
    <out_dir>/speedups_from_{P|E}_<ghz>GHz/
        speedups_{P|E}_<ghz>GHz_<bench>_phase<ph>.csv

CSV columns (matches data_loader.load_phase_data format):
    sample_index, Time_{P|E}_<src>GHz,
    Speedup_{P|E}_<tgt1>GHz_vs_{P|E}_<src>GHz,
    Speedup_{P|E}_<tgt2>GHz_vs_{P|E}_<src>GHz,
    Speedup_{P|E}_<tgt3>GHz_vs_{P|E}_<src>GHz

MAPE vs oracle speedup columns is printed per workload for validation.

Usage:
    python cross_freq_precompute.py \
        --model_base_dir results/cross_platform/cross_freq/x86_10M \
        --pmu_dir processed_data_10M/x86_desktop_heterogeneous \
        --oracle_dir results/scheduling/speedup_full/granular_phase_traces \
        --out_dir results/scheduling/cross_freq_predictions \
        --core_type P \
        --suites spec2017 spec2026 dacapo
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "cross_platform_prediction"))
from shared_features import build_features, try_load_model

FREQS = [1.0, 2.0, 3.0, 4.0]
_CPU_ID = {'P': 'cpu0', 'E': 'cpu16'}


def make_feature_args():
    import argparse
    ns = argparse.Namespace(
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


def find_phases(pmu_dir, bench, src_freq, cpu_id):
    """Return dict mapping phase index -> file Path for (bench, src_freq)."""
    pat = f"aligned_{bench}_{src_freq:.1f}GHz_{cpu_id}_phase*.csv"
    files = sorted(Path(pmu_dir).rglob(pat))
    phase_map = {}
    for f in files:
        ph_str = f.stem.rsplit("phase", 1)[-1]
        phase_map[int(ph_str)] = f
    return phase_map


def load_oracle_speedups(oracle_dir, bench, ph, src_freq, prefix):
    """Return oracle speedup dict {tgt_cfg: speedup_series} from the oracle CSV."""
    fname = f"speedups_{prefix}_{src_freq:.1f}GHz_{bench}_phase{ph}.csv"
    fpath = Path(oracle_dir) / fname
    if not fpath.exists():
        return None, None
    df = pd.read_csv(fpath)
    time_col = f"Time_{prefix}_{src_freq:.1f}GHz"
    if time_col not in df.columns:
        return None, None
    oracle_time = df[time_col].values
    speedups = {}
    for col in df.columns:
        if col.startswith(f"Speedup_{prefix}_") and f"_vs_{prefix}_" in col:
            tgt = col.split("_vs_")[0].replace("Speedup_", "")
            speedups[tgt] = df[col].values
    return oracle_time, speedups


def run_precompute(model_base_dir, pmu_dir, oracle_dir, out_dir, core_type='P',
                   suites=None):
    if suites is None:
        suites = ['spec2017']
    feat_args = make_feature_args()
    model_base_dir = Path(model_base_dir)
    pmu_dir = Path(pmu_dir)
    oracle_dir = Path(oracle_dir)
    out_dir = Path(out_dir)
    prefix = core_type          # 'P' or 'E'
    cpu_id = _CPU_ID[core_type] # 'cpu0' or 'cpu16'

    bench_to_model_dir = {}
    for suite in suites:
        suite_model_dir = model_base_dir / cpu_id / suite / "full"
        if not suite_model_dir.exists():
            print(f"[WARN] Model dir not found for suite '{suite}': {suite_model_dir}")
            continue
        suite_benches = find_workloads(suite_model_dir)
        print(f"Found {len(suite_benches)} workloads for {suite}: {suite_benches[:3]}...")
        for b in suite_benches:
            bench_to_model_dir[b] = suite_model_dir

    if not bench_to_model_dir:
        print("No workloads found across any suite.")
        return

    all_mape = []

    for bench, model_dir in bench_to_model_dir.items():
        bench_mapes = []
        for src_freq in FREQS:
            tgt_freqs = [f for f in FREQS if f != src_freq]
            src_ghz = f"{src_freq:.1f}GHz"
            out_subdir = out_dir / f"speedups_from_{prefix}_{src_ghz}"
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

            phase_map = find_phases(pmu_dir, bench, src_freq, cpu_id)
            if not phase_map:
                print(f"  [WARN] No aligned CSVs for {bench} @ {src_ghz}")
                continue

            for ph, pmu_file in sorted(phase_map.items()):
                try:
                    df_pmu = pd.read_csv(pmu_file)
                except Exception as e:
                    print(f"  [WARN] Could not read {pmu_file}: {e}")
                    continue

                X = build_features(df_pmu, "", feat_args)
                if X.empty:
                    print(f"  [WARN] Empty features for {bench} {src_ghz} phase{ph}")
                    continue

                n = len(X)

                # Oracle data for this source freq + phase
                oracle_time, oracle_speedups = load_oracle_speedups(
                    oracle_dir, bench, ph, src_freq, prefix
                )
                if oracle_time is None:
                    print(f"  [WARN] No oracle CSV for {bench} {src_ghz} phase{ph}")
                    continue

                # Truncate to shortest length (model rows may differ from oracle)
                n_out = min(n, len(oracle_time))

                out_rows = {"sample_index": np.arange(n_out)}
                out_rows[f"Time_{prefix}_{src_ghz}"] = oracle_time[:n_out]

                for tgt_freq, model in models.items():
                    tgt_ghz = f"{tgt_freq:.1f}GHz"
                    col = f"Speedup_{prefix}_{tgt_ghz}_vs_{prefix}_{src_ghz}"

                    # Model predicts log(ref_cycles_tgt / ref_cycles_src) = log(time_tgt/time_src)
                    # Speedup = time_src/time_tgt = 1 / exp(model.predict(X))
                    pred_time_ratio = np.exp(model.predict(X.iloc[:n_out]))
                    pred_speedup = 1.0 / pred_time_ratio
                    out_rows[col] = pred_speedup

                    # MAPE vs oracle
                    oracle_key = f"{prefix}_{tgt_ghz}"
                    if oracle_speedups and oracle_key in oracle_speedups:
                        y_true = oracle_speedups[oracle_key][:n_out]
                        mask = y_true > 0
                        if mask.sum() > 0:
                            mape = np.mean(np.abs(y_true[mask] - pred_speedup[mask])
                                           / (y_true[mask] + 1e-9)) * 100
                            bench_mapes.append(mape)

                out_csv = out_subdir / f"speedups_{prefix}_{src_ghz}_{bench}_phase{ph}.csv"
                pd.DataFrame(out_rows).to_csv(out_csv, index=False)

        if bench_mapes:
            mean_mape = np.mean(bench_mapes)
            print(f"  {bench}: MAPE={mean_mape:.1f}% over {len(bench_mapes)} (wl,phase,pair) combos")
            all_mape.extend(bench_mapes)

    if all_mape:
        print(f"\nOverall MAPE: {np.mean(all_mape):.2f}% (median {np.median(all_mape):.2f}%)")
    print("Done.")


_ALL_SUITES = ['spec_2017', 'spec_2026', 'dacapo_c2']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_base_dir", required=True,
                        help="e.g. results/cross_platform/cross_freq/x86_10M")
    parser.add_argument("--pmu_dir",    required=True,
                        help="e.g. processed_data_10M/x86_desktop_heterogeneous")
    parser.add_argument("--oracle_dir", required=True,
                        help="e.g. results/scheduling/speedup_full/granular_phase_traces")
    parser.add_argument("--out_dir",    required=True,
                        help="e.g. results/scheduling/cross_freq_predictions")
    parser.add_argument("--core_type",  default='P', choices=['P', 'E'],
                        help="Core type: P (cpu0, default) or E (cpu16)")
    parser.add_argument("--suites", nargs='+', default=_ALL_SUITES,
                        help=f"Benchmark suites to process (default: {' '.join(_ALL_SUITES)})")
    args = parser.parse_args()
    run_precompute(args.model_base_dir, args.pmu_dir, args.oracle_dir, args.out_dir,
                   args.core_type, args.suites)


if __name__ == "__main__":
    main()

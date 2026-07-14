"""
cross_proc_precompute.py
========================
Offline inference pass for cross-processor CatBoost models (P<->E migrations).

For each workload + phase across the specified suites:
  P-core sources (cpu0): loads aligned PMU CSVs and predicts to all 4 E-core
    frequencies via cpu0_to_cpu16 models.
  E-core sources (cpu16): loads aligned PMU CSVs and predicts to all 4 P-core
    frequencies via cpu16_to_cpu0 models.

Supports multiple benchmark suites (spec2017, spec2026, dacapo) via the
--suites flag.  Models for each suite are expected under:
    <model_dir>/cpu0_to_cpu16/<suite>/full/
    <model_dir>/cpu16_to_cpu0/<suite>/full/

Output: one CSV per (workload, phase, src_core, src_freq) in
    <out_dir>/speedups_from_{P|E}_<ghz>GHz/
        {bench}_phase<ph>.csv

CSV columns:
  P-source CSVs:  Time_P_{src}GHz, Speedup_E_{tgt}GHz_vs_P_{src}GHz (x4 targets)
  E-source CSVs:  Time_E_{src}GHz, Speedup_P_{tgt}GHz_vs_E_{src}GHz (x4 targets)

MAPE vs oracle cross-proc speedup columns is printed per workload for validation.

Usage:
    python cross_proc_precompute.py \\
        --model_dir results/cross_platform/cross_proc/x86_10M \\
        --pmu_dir processed_data_10M/x86_desktop_heterogeneous \\
        --oracle_dir results/scheduling/speedup_full/granular_phase_traces \\
        --out_dir results/scheduling/cross_proc_predictions \\
        --suites spec2017 spec2026 dacapo
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "cross_platform_prediction"))
from shared_features import build_features, try_load_model, restrict_input_counters

FREQS = [1.0, 2.0, 3.0, 4.0]
ARM_FREQS = [1.0]

_CPU_ID = {'P': 'cpu0', 'E': 'cpu16', 'L': 'cpu1', 'B': 'cpu4'}


def make_feature_args():
    return argparse.Namespace(
        input_counters=None,
    )


_SUITE_PREFIXES = {
    'spec2017': lambda b: b.startswith('spec_5'),
    'spec2026': lambda b: b.startswith('spec_') and not b.startswith('spec_5'),
    'dacapo':   lambda b: b.startswith('dacapo_'),
}


def find_workloads(p_to_e_model_dir, pmu_dir=None, suite=None):
    """Discover bench names from P->E model CBM files in the reference pair dir.

    If a single 'model_all_workloads.cbm' is found (--no_holdout training),
    discovers actual bench names from aligned CSVs in pmu_dir instead,
    filtered to the given suite.
    """
    ref_dir = Path(p_to_e_model_dir) / "cpu0_1.0GHz_to_cpu16_1.0GHz"
    benches = sorted(p.stem.replace("model_", "") for p in ref_dir.glob("model_*.cbm"))
    if benches == ['all_workloads'] and pmu_dir is not None:
        import re
        pat = re.compile(r"aligned_(.+?)_[\d.]+GHz_cpu0_phase\d+\.csv")
        discovered = set()
        for f in Path(pmu_dir).glob("aligned_*_cpu0_phase*.csv"):
            m = pat.match(f.name)
            if m:
                discovered.add(m.group(1))
        if suite and suite in _SUITE_PREFIXES:
            discovered = {b for b in discovered if _SUITE_PREFIXES[suite](b)}
        benches = sorted(discovered)
    return benches


def find_phases(pmu_dir, bench, freq, core_suffix):
    """Return dict mapping phase index -> file Path for (bench, freq, core_suffix)."""
    phase_map = {}
    for f in sorted(Path(pmu_dir).rglob(
            f"aligned_{bench}_{freq:.1f}GHz_{core_suffix}_phase*.csv")):
        phase_map[int(f.stem.rsplit("phase", 1)[-1])] = f
    return phase_map


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
                       pmu_dir, oracle_dir, out_dir, feat_args,
                       max_error_pct=None, input_counters=None):
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

        phase_map = find_phases(pmu_dir, bench, src_freq, core_suffix[0])
        for ph, pmu_file in sorted(phase_map.items()):
            try:
                df_pmu = pd.read_csv(pmu_file)
            except Exception as e:
                print(f"  [WARN] {pmu_file.name}: {e}")
                continue

            if input_counters:
                df_pmu = restrict_input_counters(df_pmu, "", input_counters)
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
                pred_speedup = np.clip(1.0 / np.exp(model.predict(X.iloc[:n_out])), 1.0/15.0, 15.0)

                oracle_key = f"{tgt_core}_{tgt_ghz}"
                if oracle_speedups and oracle_key in oracle_speedups:
                    y_true = oracle_speedups[oracle_key][:n_out]

                    if max_error_pct is not None:
                        max_frac = max_error_pct / 100.0
                        ratio = pred_speedup / (y_true + 1e-9)
                        pred_speedup = np.clip(ratio, 1 - max_frac, 1 + max_frac) * y_true

                    mask = y_true > 0
                    if mask.sum() > 0:
                        mape = np.mean(
                            np.abs(y_true[mask] - pred_speedup[mask])
                            / (np.abs(y_true[mask]) + 1e-9)
                        ) * 100
                        mapes.append(mape)

                out_rows[col] = pred_speedup

            out_csv = out_subdir / f"{bench}_phase{ph}.csv"
            pd.DataFrame(out_rows).to_csv(out_csv, index=False)

    return mapes


_ALL_SUITES = ['spec_2017', 'spec_2026', 'dacapo_c2']


def run_precompute(model_dir, pmu_dir, oracle_dir, out_dir, suites=None,
                   max_error_pct=None, input_counters=None):
    if suites is None:
        suites = _ALL_SUITES
    if max_error_pct is not None:
        print(f"Error capping enabled: predictions clamped to ±{max_error_pct}% of oracle")
    feat_args = make_feature_args()
    feat_args.input_counters = input_counters
    model_dir = Path(model_dir)
    pmu_dir = Path(pmu_dir)
    oracle_dir = Path(oracle_dir)
    out_dir = Path(out_dir)

    if arch == 'arm_edge':
        freqs = ARM_FREQS
        big_type, little_type = 'B', 'L'
        big_cpu, little_cpu = _CPU_ID['B'], _CPU_ID['L']
    else:
        freqs = FREQS
        big_type, little_type = 'P', 'E'
        big_cpu, little_cpu = _CPU_ID['P'], _CPU_ID['E']

    big_to_little_subdir = f"{big_cpu}_to_{little_cpu}"
    little_to_big_subdir = f"{little_cpu}_to_{big_cpu}"

    bench_to_dirs = {}
    for suite in suites:
        b2l_dir = model_dir / big_to_little_subdir / suite / "full"
        l2b_dir = model_dir / little_to_big_subdir / suite / "full"
        if not b2l_dir.exists():
            print(f"[WARN] {big_type}->{little_type} model dir not found for suite '{suite}': {b2l_dir}")
            continue
        if not l2b_dir.exists():
            print(f"[WARN] {little_type}->{big_type} model dir not found for suite '{suite}': {l2b_dir}")
            continue
        suite_benches = find_workloads(p_to_e_dir, pmu_dir, suite)
        print(f"Found {len(suite_benches)} workloads for {suite}: {suite_benches[:3]}...")
        for b in suite_benches:
            bench_to_dirs[b] = (b2l_dir, l2b_dir)

    if not bench_to_dirs:
        print("No workloads found across any suite.")
        return

    all_mape = []
    for bench, (b2l_dir, l2b_dir) in bench_to_dirs.items():
        bench_mapes = []

        bench_mapes += _process_direction(
            bench, FREQS, 'P', 'E', p_to_e_dir, ('cpu0', 'cpu16'),
            pmu_dir, oracle_dir, out_dir, feat_args, max_error_pct, input_counters)

        bench_mapes += _process_direction(
            bench, FREQS, 'E', 'P', e_to_p_dir, ('cpu16', 'cpu0'),
            pmu_dir, oracle_dir, out_dir, feat_args, max_error_pct, input_counters)

        if bench_mapes:
            print(f"  {bench}: MAPE={np.mean(bench_mapes):.1f}% ({len(bench_mapes)} combos)")
            all_mape.extend(bench_mapes)

    if all_mape:
        print(f"\nOverall MAPE: {np.mean(all_mape):.2f}% (median {np.median(all_mape):.2f}%)")
    print("Done.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir", required=True,
                        help="e.g. results/cross_platform/cross_proc/x86_10M")
    parser.add_argument("--pmu_dir", required=True,
                        help="e.g. processed_data_10M/x86_desktop_heterogeneous")
    parser.add_argument("--oracle_dir", required=True,
                        help="e.g. results/scheduling/speedup_full/granular_phase_traces")
    parser.add_argument("--out_dir", required=True,
                        help="e.g. results/scheduling/cross_proc_predictions")
    parser.add_argument("--suites", nargs='+', default=_ALL_SUITES,
                        help=f"Benchmark suites to process (default: {' '.join(_ALL_SUITES)})")
    parser.add_argument("--max_error_pct", type=float, default=None,
                        help="Cap per-sample prediction error to ±X%% of oracle. "
                             "Sweep this to test model accuracy requirements.")
    parser.add_argument("--input_counters", nargs='+', default=None,
                        help="Must match the --input_counters used during training. "
                             "Required when models were trained with a counter subset.")
    args = parser.parse_args()
    run_precompute(args.model_dir, args.pmu_dir, args.oracle_dir, args.out_dir,
                   args.suites, args.max_error_pct, args.input_counters)

if __name__ == "__main__":
    main()

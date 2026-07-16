#!/usr/bin/env python3
"""
analyze_phase_forecast.py
=========================
Summarize phase_forecast_10M.csv:

  1. Per-method overall MAPE (mean over benchmarks): global vs per_phase vs
     conditioned vs persistence, and how often per_phase beats global.
  2. Per-benchmark global-vs-per_phase deltas.
  3. Per-phase breakdown: does per-phase forecasting help most on high-CoV
     (volatile) phases? Aggregates the phase-level rows by CoV bucket.
"""
import os
import re
import argparse
import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
DEFAULT_CSV = os.path.join(REPO_ROOT, "results/forecasting/phase_forecasting/phase_forecast_10M.csv")


def suite_of(bench):
    """dacapo vs spec2017 vs spec2026 from the benchmark name. DaCapo (JVM: JIT +
    recurring GC phases) and SPEC (monolithic compute) behave very differently, so
    the per-phase benefit and persistence strength are reported per suite to avoid
    conflating suite characteristics with the phase-forecasting effect."""
    b = bench.replace("aligned_", "")
    if b.startswith("dacapo"):
        return "dacapo"
    m = re.match(r"spec_(\d+)", b)
    if m:
        return "spec2026" if int(m.group(1)) >= 700 else "spec2017"
    return "other"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=DEFAULT_CSV)
    args = ap.parse_args()
    if not os.path.exists(args.csv):
        print(f"No CSV at {args.csv} — run ./run_phase_forecast.sh first.")
        return
    d = pd.read_csv(args.csv)

    overall = d[d.phase == "all"].copy()
    overall["suite"] = overall.bench.map(suite_of)
    perbench = overall.pivot_table(index="bench", columns="method", values="mape")
    suite_by_bench = overall.drop_duplicates("bench").set_index("bench")["suite"]
    methods = [m for m in ["global", "per_phase", "per_phase_pred", "per_phase_oracle_next",
                            "per_phase_gated", "conditioned", "persistence"]
               if m in perbench.columns]

    print(f"=== Per-method overall MAPE (mean over {perbench.shape[0]} benches) ===")
    print(f"  {'method':13s} {'MAPE':>7}")
    for m in methods:
        print(f"  {m:13s} {perbench[m].mean():7.2f}")

    # Per-suite: DaCapo (phasey JVM) vs SPEC (monolithic) may differ sharply.
    print("\n=== Per-method MAPE by suite (mean over that suite's benches) ===")
    hdr = "  ".join(f"{m:>11}" for m in methods)
    print(f"  {'suite':10s} {'n':>3}  {hdr}   pp_wins")
    for suite, benches in suite_by_bench.groupby(suite_by_bench).groups.items():
        sub = perbench.loc[list(benches)]
        cells = "  ".join(f"{sub[m].mean():11.2f}" for m in methods)
        wins = ""
        if "per_phase" in sub and "global" in sub:
            dlt = sub["per_phase"] - sub["global"]
            wins = f"{int((dlt < 0).sum())}/{len(dlt)}"
        print(f"  {suite:10s} {len(benches):>3}  {cells}   {wins}")

    if "per_phase" in perbench and "global" in perbench:
        delta = perbench["per_phase"] - perbench["global"]
        wins = int((delta < 0).sum())
        print(f"\n  per_phase beats global on {wins}/{len(delta)} benches; "
              f"mean Δ(per_phase-global) = {delta.mean():+.2f} MAPE")
        print("\n=== Per-benchmark: global vs per_phase ===")
        print(f"  {'bench':32s} {'global':>7} {'per_phase':>10} {'Δ':>7}")
        for b in perbench.index:
            g, p = perbench.loc[b, "global"], perbench.loc[b, "per_phase"]
            print(f"  {b:32s} {g:7.2f} {p:10.2f} {p - g:+7.2f}")

    # Per-phase breakdown: pair global vs per_phase at the phase level, bucket by CoV.
    ph = d[d.phase != "all"].copy()
    if not ph.empty:
        ph["phase"] = ph["phase"].astype(int)
        key = ["bench", "phase"]
        g = ph[ph.method == "global"].set_index(key)[["mape", "n", "cov"]]
        p = ph[ph.method == "per_phase"].set_index(key)[["mape"]].rename(columns={"mape": "mape_pp"})
        j = g.join(p, how="inner").dropna()
        j = j[j.n >= 5]   # ignore tiny (1-4 window) phases
        if not j.empty:
            j["delta"] = j["mape_pp"] - j["mape"]
            j["cov_bucket"] = pd.cut(j["cov"], [0, 0.1, 0.25, 1e9],
                                     labels=["low(<0.10)", "mid(0.10-0.25)", "high(>0.25)"])
            print("\n=== Per-phase (n>=5): does per-phase help more on volatile phases? ===")
            print(f"  {'CoV bucket':16s} {'#phases':>8} {'global':>7} {'per_phase':>10} {'Δ':>7}")
            for bucket, grp in j.groupby("cov_bucket", observed=True):
                print(f"  {str(bucket):16s} {len(grp):8d} {grp['mape'].mean():7.2f} "
                      f"{grp['mape_pp'].mean():10.2f} {grp['delta'].mean():+7.2f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
analyze_het_phase.py
====================
Compare heterogeneous-history regimes (homogeneous / naive / translated) for
phase-aware forecasting. For each method, shows @all and @transition MAPE per
regime, and how close naive/translated land to the homogeneous floor.
"""
import os
import argparse
import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
METHODS = ["global", "per_phase", "per_phase_gated", "persistence"]
REGIMES = ["homogeneous", "naive", "translated"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True)
    args = ap.parse_args()
    d = pd.read_csv(args.csv)
    if "regime" not in d.columns:
        print("CSV lacks a 'regime' column — re-run with the updated harness.")
        return

    n_bench = d.bench.nunique()
    mode = d.het_mode[d.het_mode != "none"].iloc[0] if (d.het_mode != "none").any() else "?"
    prob = d.het_prob.max()
    print(f"=== Heterogeneous-history phase forecasting ({n_bench} benches, "
          f"mode={mode}, prob={prob}) ===")

    for ph in ["all", "transition"]:
        sub = d[d.phase == ph]
        piv = sub.pivot_table(index="regime", columns="method", values="mape")
        piv = piv.reindex(index=[r for r in REGIMES if r in piv.index])
        cols = [m for m in METHODS if m in piv.columns]
        print(f"\n  @{ph} (mean MAPE over benches)")
        print(f"    {'regime':13s} " + "  ".join(f"{m:>16}" for m in cols))
        for r in piv.index:
            print(f"    {r:13s} " + "  ".join(f"{piv.loc[r, m]:16.2f}" for m in cols))

    # recovery: how much of the naive->homogeneous gap does translation close?
    piv = d[d.phase == "all"].pivot_table(index="regime", columns="method", values="mape")
    if all(r in piv.index for r in REGIMES):
        print("\n  Translation recovery (@all, per method):")
        print(f"    {'method':18s} {'homog':>8} {'naive':>8} {'transl':>8}  {'recovered':>10}")
        for m in [c for c in METHODS if c in piv.columns]:
            h, n, t = piv.loc["homogeneous", m], piv.loc["naive", m], piv.loc["translated", m]
            rec = (n - t) / (n - h) * 100 if (n - h) != 0 else float("nan")
            print(f"    {m:18s} {h:8.2f} {n:8.2f} {t:8.2f}  {rec:9.1f}%")


if __name__ == "__main__":
    main()

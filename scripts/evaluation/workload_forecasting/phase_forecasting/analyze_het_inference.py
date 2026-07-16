#!/usr/bin/env python3
"""Summarize het-inference: per (direction, regime) MAPE for each method vs the
honest translated-persistence baseline. Highlights whether the forecaster beats
persistence in the cold (translated) case, split @all and @transition."""
import argparse
import numpy as np
import pandas as pd

METHODS = ["global", "per_phase", "per_phase_gated", "persistence"]
REGIMES = ["oracle", "translated", "naive"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    a = ap.parse_args()
    d = pd.read_csv(a.csv)
    print(f"=== Het inference ({d.bench.nunique()} benches) ===")
    for mode, dm in d.groupby("mode"):
        print(f"\n### direction = {mode}  ({dm['source'].iloc[0]}) ###")
        for ph in ["all", "transition"]:
            w = dm[dm.phase == ph].pivot_table(index="regime", columns="method", values="mape")
            w = w.reindex([r for r in REGIMES if r in w.index])
            cols = [m for m in METHODS if m in w.columns]
            print(f"  @{ph}")
            print(f"    {'regime':12s} " + "  ".join(f"{m:>16}" for m in cols) + "   fcst<persist?")
            for r in w.index:
                cells = "  ".join(f"{w.loc[r, m]:16.2f}" for m in cols)
                beat = ""
                if "persistence" in w.columns and "global" in w.columns:
                    beat = "YES" if w.loc[r, "global"] < w.loc[r, "persistence"] else "no"
                    if "per_phase_gated" in w.columns and w.loc[r, "per_phase_gated"] < w.loc[r, "persistence"]:
                        beat += "(+gated)"
                print(f"    {r:12s} {cells}   {beat}")


if __name__ == "__main__":
    main()

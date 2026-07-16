#!/usr/bin/env python3
"""
analyze_multiphase.py
=====================
Split benchmarks by how MULTI-PHASE they are, then compare the forecasting methods
within each group. Phase-aware forecasting can only help where there really are
multiple phases; a benchmark that GMM lumps into one dominant cluster is effectively
single-phase and per-phase routing must ≈ global there.

Multi-phase indicators (per benchmark, from the test-window phase distribution; phase
detection is identical across models so the split is model-independent):
  - eff_phases : effective # phases = 1 / Σ(fraction_i²)  (inverse Simpson).
                 ~1 => one dominant phase; higher => mass spread over several.
  - dom_frac   : fraction of test windows in the single largest phase.
  - trans_rate : fraction of steps where the phase changes (from phase_pred_stats).

A benchmark is "multi-phase" iff eff_phases >= --thresh (default 2.0).

Usage:
  python analyze_multiphase.py                         # all phase_forecast_10M_*_gmm.csv
  python analyze_multiphase.py --csvs a.csv b.csv
  python analyze_multiphase.py --thresh 1.5
"""
import os
import re
import glob
import argparse
import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
PHASE_DIR = os.path.join(REPO_ROOT, "results/forecasting/phase_forecasting")
METHODS = ["global", "per_phase", "per_phase_pred", "per_phase_oracle_next",
           "per_phase_gated", "persistence"]


def model_of(path):
    m = re.search(r"phase_forecast_10M_([a-z]+)_", os.path.basename(path))
    return m.group(1) if m else os.path.basename(path)


def multiphase_metrics(d):
    """Per-benchmark multi-phase indicators from the global method's per-phase counts."""
    g = d[(d["method"] == "global") & (d["phase"] != "all")].copy()
    g["phase"] = g["phase"].astype(int)
    rows = []
    for b, grp in g.groupby("bench"):
        n = grp.set_index("phase")["n"].astype(float)
        frac = n / n.sum()
        rows.append((b, len(n), float(1.0 / (frac ** 2).sum()), float(frac.max())))
    m = pd.DataFrame(rows, columns=["bench", "n_phases", "eff_phases", "dom_frac"]).set_index("bench")
    st = d[d["method"] == "phase_pred_stats"].set_index("bench")
    if len(st):
        m["trans_rate"] = st["cov"]
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csvs", nargs="+", default=None)
    ap.add_argument("--thresh", type=float, default=2.0, help="eff_phases cutoff for multi-phase")
    args = ap.parse_args()

    csvs = args.csvs or sorted(glob.glob(os.path.join(PHASE_DIR, "phase_forecast_10M_*_gmm.csv")))
    csvs = [c for c in csvs if os.path.exists(c)]
    if not csvs:
        print("No phase CSVs found — run ./run_all_models.sh first.")
        return
    data = {model_of(c): pd.read_csv(c) for c in csvs}

    # Multi-phase split (model-independent; compute from the first available CSV).
    mp = multiphase_metrics(next(iter(data.values())))
    mp["group"] = np.where(mp["eff_phases"] >= args.thresh, "multi", "single")

    print(f"=== Multi-phase indicators (thresh eff_phases>={args.thresh}) ===")
    print(f"  {'bench':34s} {'eff_ph':>6} {'dom_fr':>6} {'trans':>6}  group")
    show = mp.sort_values("eff_phases", ascending=False)
    for b, r in show.iterrows():
        tr = r.get("trans_rate", np.nan)
        print(f"  {b.replace('aligned_','').replace('_4.0GHz_cpu0',''):34s} "
              f"{r.eff_phases:6.2f} {r.dom_frac:6.2f} {tr:6.3f}  {r.group}")
    n_multi = int((mp.group == "multi").sum())
    print(f"\n  multi-phase: {n_multi}/{len(mp)}   single-phase: {len(mp) - n_multi}/{len(mp)}")

    # Per-model method means within each group.
    for model, d in data.items():
        o = d[d["phase"] == "all"].pivot_table(index="bench", columns="method", values="mape")
        o = o.join(mp["group"])
        ms = [m for m in METHODS if m in o.columns]
        print(f"\n=== {model}: method MAPE by group (mean) ===")
        print(f"  {'group':8s} {'n':>3}  " + "  ".join(f"{m:>10}" for m in ms))
        for grp in ["multi", "single"]:
            sub = o[o.group == grp]
            if sub.empty:
                continue
            cells = "  ".join(f"{sub[m].mean():10.2f}" for m in ms)
            print(f"  {grp:8s} {len(sub):>3}  {cells}")
        # per_phase vs global lift within multi-phase group
        sub = o[o.group == "multi"]
        if not sub.empty and "per_phase" in sub and "global" in sub:
            dl = sub["per_phase"] - sub["global"]
            print(f"  -> multi-phase per_phase vs global: mean {dl.mean():+.2f}, "
                  f"wins {int((dl < -0.05).sum())}/{len(dl)}")


if __name__ == "__main__":
    main()

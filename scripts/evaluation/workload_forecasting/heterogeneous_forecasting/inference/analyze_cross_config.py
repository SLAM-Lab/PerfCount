#!/usr/bin/env python3
"""
analyze_cross_config.py
=======================
Turn cross_config_10M.csv into the source->target matrices:

  "Given a heterogeneous history observed at S, unify it to each target config C
   and predict C's future. How close is that to (a) C's homogeneous forecast
   (oracle) and (b) C's ground truth?"

Every cell is MAPE against C's ground-truth future (mean over benchmarks). For a
target C, `oracle` is the floor (forecasting while actually on C); `translated`
is forecasting C from a foreign S after unification; the gap translated-oracle is
the price of being on the wrong config.

Sections printed:
  1. Category summary (cross-freq / P->E / E->P): oracle vs naive vs translated.
  2. Cross-FREQUENCY matrices (per core): src_freq x tgt_freq, translated MAPE,
     with each target's oracle shown for reference.
  3. Cross-PROCESSOR matrices (P->E and E->P): src_freq x tgt_freq.

Usage:
  python analyze_cross_config.py                      # default CSV, model dt
  python analyze_cross_config.py --model mlp
  python analyze_cross_config.py --csv /path/to.csv --metric gap
"""
import os
import argparse
import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
DEFAULT_CSV = os.path.join(REPO_ROOT, "results/forecasting/cross_config/cross_config_10M.csv")
FREQS = [1.0, 2.0, 3.0, 4.0]


def category(r):
    if (r.src_cpu, r.src_freq) == (r.tgt_cpu, r.tgt_freq):
        return "identity"
    if r.src_cpu == r.tgt_cpu:
        return "cross-freq"
    return "P->E" if r.src_cpu == 0 else "E->P"


def _matrix(sub, value_col):
    """src_freq (rows) x tgt_freq (cols) mean of value_col."""
    m = sub.pivot_table(index="src_freq", columns="tgt_freq", values=value_col, aggfunc="mean")
    return m.reindex(index=FREQS, columns=FREQS)


def _print_matrix(title, sub, oracle_by_tgt):
    print(f"\n{title}   (translated MAPE %, mean over benchmarks)")
    mat = _matrix(sub, "translated")
    hdr = "  src\\tgt " + "".join(f"{f:>8.1f}" for f in FREQS)
    print(hdr)
    for f in FREQS:
        row = mat.loc[f]
        cells = "".join((f"{row[t]:>8.1f}" if pd.notna(row[t]) else f"{'-':>8}") for t in FREQS)
        print(f"  {f:>6.1f}  {cells}")
    orc = "".join(f"{oracle_by_tgt.get(t, np.nan):>8.1f}" for t in FREQS)
    print(f"  oracle  {orc}   <- homogeneous floor per target freq")


def summarize_models(d, horizon):
    """Cross-model comparison: mean MAPE per model x method, overall and by
    direction, plus the persistence comparison. Answers 'which model is best'."""
    if horizon is not None:
        d = d[d.horizon == horizon]
    key = ["src_cpu", "src_freq", "tgt_cpu", "tgt_freq", "bench", "model"]
    w = d.pivot_table(index=key, columns="method", values="mape").reset_index()
    w["cat"] = w.apply(lambda r: "cross-core" if category(r) in ("P->E", "E->P") else category(r), axis=1)
    order = ["dt", "mlp", "lstm", "transformer"]
    models = [m for m in order if m in set(w.model)]
    hlabel = f"h={horizon}" if horizon is not None else "all horizons"
    print(f"\n=== Model comparison ({hlabel}, mean MAPE %) ===")
    print(f"  {'model':12s} {'oracle':>7} {'translated':>11} {'naive':>7} {'persist':>8}  beats_persist(oracle)")
    for m in models:
        s = w[w.model == m]
        o = s.oracle.mean(); t = s.translated.mean(); nv = s.naive.mean(); p = s.persistence.mean()
        flag = "YES +%.1f" % (p - o) if o < p else "no  -%.1f" % (o - p)
        print(f"  {m:12s} {o:7.1f} {t:11.1f} {nv:7.1f} {p:8.1f}  {flag}")
    print("\n=== by direction: oracle | translated (mean MAPE %) ===")
    print(f"  {'model':12s} {'identity':>16} {'cross-freq':>16} {'cross-core':>16}")
    for m in models:
        s = w[w.model == m]
        cells = []
        for c in ["identity", "cross-freq", "cross-core"]:
            ss = s[s.cat == c]
            cells.append(f"{ss.oracle.mean():5.1f}|{ss.translated.mean():5.1f}" if len(ss) else f"{'-':>11}")
        print(f"  {m:12s} " + "".join(f"{c:>16}" for c in cells))
    best = min(models, key=lambda m: w[w.model == m].oracle.mean())
    print(f"\n  best forecaster (lowest oracle MAPE): {best}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--model", default="dt")
    ap.add_argument("--horizon", type=int, default=None)
    ap.add_argument("--timesteps", type=int, default=None)
    ap.add_argument("--summary", action="store_true",
                    help="cross-model comparison (all models) instead of one model's matrices")
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        print(f"No CSV at {args.csv} — run ./run_cross_config.sh first.")
        return
    d = pd.read_csv(args.csv)
    if args.timesteps is not None:
        d = d[d.timesteps == args.timesteps]
    if args.summary:
        summarize_models(d, args.horizon)
        return
    d = d[d.model == args.model]
    if args.horizon is not None:
        d = d[d.horizon == args.horizon]
    if args.timesteps is not None:
        d = d[d.timesteps == args.timesteps]
    if d.empty:
        print("No rows for that filter.")
        return

    # wide: one row per (src,tgt,bench,...) with a column per method
    key = ["src_cpu", "src_freq", "tgt_cpu", "tgt_freq", "bench"]
    w = d.pivot_table(index=key, columns="method", values="mape").reset_index()
    w["cat"] = w.apply(category, axis=1)
    w["gap"] = w.translated - w.oracle

    n_bench = w.bench.nunique()
    print(f"Model={args.model}  benches={n_bench}"
          + (f"  H={args.horizon}" if args.horizon else "")
          + (f"  T={args.timesteps}" if args.timesteps else ""))

    # 1. category summary
    print("\n=== 1. Category summary (mean MAPE %) ===")
    summ = (w[w.cat != "identity"]
            .groupby("cat")[["oracle", "naive", "translated", "gap"]]
            .mean().round(1))
    summ = summ.rename(columns={"gap": "gap(tr-orc)"})
    print(summ.to_string())

    # oracle floor per target (cpu,freq) — depends only on target
    oracle_tgt = (w.groupby(["tgt_cpu", "tgt_freq"]).oracle.mean())

    # 2. cross-frequency (per core)
    print("\n=== 2. Cross-FREQUENCY (unify across clock, same core) ===")
    for cpu, label in [(0, "P-core"), (16, "E-core")]:
        sub = w[(w.cat == "cross-freq") & (w.tgt_cpu == cpu)]
        if sub.empty:
            continue
        ofl = {f: oracle_tgt.get((cpu, f), np.nan) for f in FREQS}
        _print_matrix(f"[{label}]", sub, ofl)

    # 3. cross-processor
    print("\n=== 3. Cross-PROCESSOR (unify across core) ===")
    for cat, tgt_cpu, label in [("P->E", 16, "P-core src -> E-core tgt"),
                                 ("E->P", 0, "E-core src -> P-core tgt")]:
        sub = w[w.cat == cat]
        if sub.empty:
            continue
        ofl = {f: oracle_tgt.get((tgt_cpu, f), np.nan) for f in FREQS}
        _print_matrix(f"[{label}]", sub, ofl)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Analyze the DVFS (same-core cross-frequency) forecasting sweep, split by CORE
(P-core cpu0 vs E-core cpu16). For each model CSV (dvfs_forecast_<model>_delta.csv)
reports the phase-UNAWARE forecaster (global), the phase-AWARE forecaster
(best of per_phase/per_phase_gated), and translated-persistence -- MAPE @all and
@transition, win% (forecaster beats translated-persistence), the phase-awareness
benefit (global - best phase-aware), and a frequency-distance/direction breakdown.

Usage:
  analyze_dvfs.py --glob 'results/forecasting/phase_forecasting/dvfs_forecast_*_delta.csv'
  analyze_dvfs.py --csv  results/forecasting/phase_forecasting/dvfs_forecast_dt_delta.csv
"""
import argparse, glob, os, re, numpy as np, pandas as pd

NOPHASE = re.compile(r'^aligned_(?P<rest>.+)_(?P<freq>[\d.]+)GHz_cpu(?P<cpu>\d+)$')
CORE = {'0': 'P-core (cpu0)', '16': 'E-core (cpu16)'}


def parse_bench(b):
    m = NOPHASE.match(b)
    return (m.group('cpu'), float(m.group('freq'))) if m else (None, None)


def load_wide(csv):
    """One row per (bench, source, core, fdist) with fc/ps columns @all and @transition."""
    d = pd.read_csv(csv)
    d = d[d.regime == 'translated'].copy()          # honest cold-prediction regime
    tc, tf = zip(*d.bench.map(parse_bench))
    d['tcpu'], d['tfreq'] = tc, tf
    d['sfreq'] = d.source.str.split(':').str[1].astype(float)
    d['fdist'] = (d.tfreq - d.sfreq)                 # signed: + = predict higher freq
    key = ['bench', 'source', 'tcpu', 'tfreq', 'sfreq', 'fdist']
    w = d.pivot_table(index=key, columns=['method', 'phase'], values='mape').reset_index()

    def col(meth, ph): return (meth, ph) if (meth, ph) in w.columns else None
    def get(meth, ph):
        c = col(meth, ph); return w[c] if c else pd.Series(np.nan, index=w.index)
    for ph in ['all', 'transition']:
        w[f'glob_{ph}'] = get('global', ph)          # phase-UNAWARE
        aware = [get(m, ph) for m in ('per_phase', 'per_phase_gated')]
        w[f'aware_{ph}'] = pd.concat(aware, axis=1).min(axis=1)   # best phase-AWARE
        w[f'fc_{ph}'] = pd.concat([w[f'glob_{ph}'], w[f'aware_{ph}']], axis=1).min(axis=1)
        w[f'ps_{ph}'] = get('persistence', ph)       # translated-persistence
    return w


def row(lbl, sub):
    n = len(sub)
    wa = (sub.fc_all < sub.ps_all).mean() * 100
    wt = (sub.fc_transition < sub.ps_transition).mean() * 100
    ben = (sub.glob_all - sub.aware_all).mean()      # + => phase-awareness helps
    return (f"  {lbl:16s} {n:>5} {sub.glob_all.mean():7.1f} {sub.aware_all.mean():7.1f} "
            f"{sub.ps_all.mean():7.1f} {wa:4.0f}%  {sub.fc_transition.mean():7.1f} "
            f"{sub.ps_transition.mean():7.1f} {wt:4.0f}%  {ben:+6.2f}")


HDR = (f"  {'group':16s} {'n':>5} {'glob':>7} {'aware':>7} {'persist':>7} {'win%':>5}  "
       f"{'fc@tr':>7} {'ps@tr':>7} {'win%':>5}  {'awareΔ':>6}")


def report(csv):
    w = load_wide(csv)
    model = re.sub(r'^dvfs_forecast_|_delta\.csv$', '', os.path.basename(csv))
    print(f"\n{'='*96}\n=== DVFS forecasting : model={model}  ({w.bench.nunique()} benches, "
          f"{len(w)} bench-pairs) ===")
    print("    glob=phase-unaware  aware=best phase-aware  persist=translated-persistence  "
          "awareΔ=glob-aware(@all)")
    for cpu, name in CORE.items():
        wc = w[w.tcpu == cpu]
        if not len(wc):
            continue
        print(f"\n--- {name} ---\n{HDR}")
        print(row('ALL pairs', wc))
        for lbl, sub in [('+1 (speed up)', wc[wc.fdist == 1]), ('+2', wc[wc.fdist == 2]),
                         ('+3', wc[wc.fdist == 3]), ('-1 (slow down)', wc[wc.fdist == -1]),
                         ('-2', wc[wc.fdist == -2]), ('-3', wc[wc.fdist == -3])]:
            if len(sub):
                print(row(lbl, sub))
    print(f"\n--- both cores combined ---\n{HDR}")
    print(row('ALL pairs', w))
    print(row('up-clock (+)', w[w.fdist > 0]))
    print(row('down-clock (-)', w[w.fdist < 0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv')
    ap.add_argument('--glob', default=None)
    a = ap.parse_args()
    files = sorted(glob.glob(a.glob)) if a.glob else [a.csv]
    files = [f for f in files if f and os.path.exists(f)]
    if not files:
        raise SystemExit("no input CSVs found")
    for f in files:
        report(f)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# Cross-study DVFS taxonomy: for each temporal approach (Reactive | Gated Forecast | Perfect
# Future), show performance across the translation-model-quality ladder
#   Heuristic (governor) -> LOOCV -> General -> Perfect-CP
# The model rungs (LOOCV/General/Perfect-CP) come from the four separate model studies; the
# Heuristic rung is the best governor operating in that temporal mode (reactive / forecast /
# future), read from the LOOCV study. Normalized to the per-core Viterbi oracle.
#
# Usage:  RES=<results/scheduling> python cross_taxonomy.py [out_dir]
import os, sys, re
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

RES = os.environ.get('RES', 'results/scheduling')
OUT = sys.argv[1] if len(sys.argv) > 1 else f'{RES}/DVFS/cross_taxonomy'
os.makedirs(OUT, exist_ok=True)

# model-quality rung -> DVFS study dir
STUDIES = {'LOOCV': 'loocv', 'General': 'gentemporal', 'Perfect-CP': 'oracle'}
RUNGS = ['Heuristic', 'LOOCV', 'General', 'Perfect-CP']

# governor families per temporal mode (best = min ratio)
def govs(mode, core):
    base = ['Ondemand', 'Conservative', 'Schedutil', 'Interactive', 'Intel_HWP', 'EWMA', 'UCB1']
    if mode == 'reactive':   return [f'{g}_{core}' if g != 'Schedutil' else f'Schedutil_PELT_{core}' for g in base]
    suffix = {'forecast': 'Forecast', 'future': 'Future'}[mode]
    return [f'{g}_{suffix}_{core}' for g in base]

# model policy per temporal group
MODEL = {'Reactive': 'Model_Greedy_{c}', 'Gated Forecast': 'Model_Forecast_ReactiveGated_{c}',
         'Perfect Future': 'Model_Greedy_Oracle_{c}'}
GOV_MODE = {'Reactive': 'reactive', 'Gated Forecast': 'forecast', 'Perfect Future': 'future'}

def suite_of(w):
    if w.startswith('dacapo'): return 'DaCapo'
    m = re.match(r'spec_(\d+)', w)
    if not m: return 'other'
    return 'SPEC2017' if int(m.group(1)) < 700 else 'SPEC2026'


def load(study, metric, suite=None):
    p = f'{RES}/DVFS/{study}/all_phases_summary.csv'
    if not os.path.exists(p): return None
    E = pd.read_csv(p); E = E[E.Metric == metric]
    if suite is not None:
        E = E[E.Workload.map(suite_of) == suite]
    if E.empty: return None
    return E.pivot_table(index=['Workload', 'Phase'], columns='Policy', values='Final_Value')

def ratio(E, pol, norm):
    return (E[pol] / E[norm]).mean() if (E is not None and pol in E and norm in E) else np.nan

for core in ['P', 'E']:
  norm = f'Global_Oracle_{core}'
  for metric in ['EDP', 'ED2P']:
    for suite in [None, 'SPEC2017', 'SPEC2026', 'DaCapo']:
        loo = load('loocv', metric, suite)
        if loo is None: continue
        groups = list(MODEL.keys())
        vals = {g: [] for g in groups}
        for g in groups:
            # Heuristic rung: best governor in this temporal mode, from the LOOCV study
            gr = [ratio(loo, p, norm) for p in govs(GOV_MODE[g], core)]
            gr = [x for x in gr if x == x]
            vals[g].append(min(gr) if gr else np.nan)
            # model rungs
            for rung in ['LOOCV', 'General', 'Perfect-CP']:
                E = load(STUDIES[rung], metric, suite)
                vals[g].append(ratio(E, MODEL[g].format(c=core), norm))

        # grouped bar chart
        fig, ax = plt.subplots(figsize=(11, 6))
        x = np.arange(len(groups)); w = 0.19
        colors = ['#9e9e9e', '#4e79a7', '#59a14f', '#e15759']  # heuristic, loocv, general, perfect
        for j, rung in enumerate(RUNGS):
            heights = [vals[g][j] for g in groups]
            bars = ax.bar(x + (j - 1.5) * w, heights, w, label=rung, color=colors[j], edgecolor='black', linewidth=0.4)
            for b, h in zip(bars, heights):
                if h == h: ax.text(b.get_x() + b.get_width()/2, h + 0.01, f'{h:.3f}', ha='center', va='bottom', fontsize=7)
        ax.axhline(1.0, color='red', ls='--', lw=1.2, label='Oracle = 1.0')
        ax.set_xticks(x); ax.set_xticklabels(groups)
        ax.set_ylabel(f'Norm. {metric}  (policy / {norm}, lower = better)')
        ax.set_title(f'{core}-core DVFS {suite or "all suites"} — '
                     f'temporal approach x model quality  ({metric})')
        ax.legend(ncol=5, fontsize=9, loc='upper right')
        ax.set_ylim(0.9, max(1.05, np.nanmax([v for g in groups for v in vals[g]]) * 1.08))
        fig.tight_layout()
        tag = '' if suite is None else f'_{suite}'
        f = f'{OUT}/{core}_DVFS_{metric}{tag}_cross_taxonomy.png'
        fig.savefig(f, dpi=130); plt.close(fig)
        print(f'wrote {f}')

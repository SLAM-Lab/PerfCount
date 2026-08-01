#!/usr/bin/env python3
# Cross-study HETEROGENEOUS taxonomy: temporal approach (Reactive | Gated Forecast | Perfect
# Future) x translation-model quality (Heuristic | LOOCV | General | Perfect-CP), per suite.
# Model rungs come from the four cross-processor model studies; the Heuristic rung is the best
# heuristic operating in that temporal mode (hetero heuristics have reactive and perfect-future
# variants only, so the forecast group reuses the reactive heuristics). Normalized to the
# full-trace Viterbi global oracle.
#
# Usage:  RES=<results/scheduling> python cross_taxonomy_hetero.py [out_dir]
import os, sys, re
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

RES = os.environ.get('RES', 'results/scheduling')
OUT = sys.argv[1] if len(sys.argv) > 1 else f'{RES}/hetero/cross_taxonomy'
os.makedirs(OUT, exist_ok=True)

# Cross-processor model quality, holding everything else fixed. All three arms use the
# gentemporal cross-FREQUENCY half and the corrected gate (which now pays the cross-cluster
# warmup penalty), so the only thing varying across them is the cross-processor model.
# The pre-2026-07-23 dirs (loocv, gentemporal, oracle) predate that fix and pinned the
# cross-frequency half to top4, so they are not comparable to each other or to these.
STUDIES = {'LOOCV': 'loocv_fixed', 'General': 'gt_baseline', 'Perfect-CP': 'perfectcp_fixed'}
RUNGS = ['Heuristic', 'LOOCV', 'General', 'Perfect-CP']
NORM = 'Proactive_Hetero_Oracle'

MODEL = {'Reactive': 'Model_Reactive_Hetero',
         'Gated Forecast': 'Model_Forecast_ReactiveGated_Hetero',
         'Perfect Future': 'Model_Greedy_Oracle_Hetero'}

HEUR_REACTIVE = ['EAS_Hetero', 'EAS_With_DVFS', 'Micro_EAS',
                 'Thread_Director', 'Threshold_Migration', 'UCB1_Hetero']
HEUR_FUTURE = ['EAS_Oracle_Hetero', 'Thread_Director_Oracle']
# hetero heuristics have no forecast-fed variant; the forecast group reuses the reactive rules
HEURS = {'Reactive': HEUR_REACTIVE, 'Gated Forecast': HEUR_REACTIVE, 'Perfect Future': HEUR_FUTURE}


def suite_of(w):
    if w.startswith('dacapo'):
        return 'DaCapo'
    m = re.match(r'spec_(\d+)', w)
    if not m:
        return 'other'
    return 'SPEC2017' if int(m.group(1)) < 700 else 'SPEC2026'


def load(study, metric, suite):
    p = f'{RES}/hetero/{study}/all_phases_summary.csv'
    if not os.path.exists(p):
        return None
    E = pd.read_csv(p)
    E = E[E.Metric == metric]
    E = E[E.Workload.map(suite_of) == suite]
    if E.empty:
        return None
    return E.pivot_table(index=['Workload', 'Phase'], columns='Policy', values='Final_Value')


def ratio(E, pol):
    return (E[pol] / E[NORM]).mean() if (E is not None and pol in E and NORM in E) else np.nan


for suite in ['SPEC2017', 'SPEC2026', 'DaCapo']:
    for metric in ['EDP', 'ED2P']:
        groups = list(MODEL.keys())
        vals = {g: [] for g in groups}
        loo = load('loocv', metric, suite)
        for g in groups:
            hr = [ratio(loo, h) for h in HEURS[g]]
            hr = [x for x in hr if x == x]
            vals[g].append(min(hr) if hr else np.nan)
            for rung in ['LOOCV', 'General', 'Perfect-CP']:
                vals[g].append(ratio(load(STUDIES[rung], metric, suite), MODEL[g]))

        fig, ax = plt.subplots(figsize=(11, 6))
        x = np.arange(len(groups)); w = 0.19
        colors = ['#9e9e9e', '#4e79a7', '#59a14f', '#e15759']
        for j, rung in enumerate(RUNGS):
            h = [vals[g][j] for g in groups]
            bars = ax.bar(x + (j - 1.5) * w, h, w, label=rung, color=colors[j],
                          edgecolor='black', linewidth=0.4)
            for b, v in zip(bars, h):
                if v == v:
                    ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f'{v:.3f}',
                            ha='center', va='bottom', fontsize=7)
        ax.axhline(1.0, color='red', ls='--', lw=1.2, label='Oracle = 1.0')
        ax.set_xticks(x); ax.set_xticklabels(groups)
        ax.set_ylabel(f'Norm. {metric}  (policy / global oracle, lower = better)')
        ax.set_title(f'Heterogeneous {suite} — temporal approach x model quality  ({metric})')
        ax.legend(ncol=5, fontsize=9, loc='upper right')
        finite = [v for g in groups for v in vals[g] if v == v]
        ax.set_ylim(0.9, max(1.05, (max(finite) if finite else 1.0) * 1.10))
        fig.tight_layout()
        f = f'{OUT}/Hetero_{suite}_{metric}_cross_taxonomy.png'
        fig.savefig(f, dpi=130); plt.close(fig)
        print(f'wrote {f}')

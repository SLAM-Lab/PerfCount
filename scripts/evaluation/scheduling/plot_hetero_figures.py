#!/usr/bin/env python3
"""Chapter 5 heterogeneous figures.

  fig_het_reactive.pdf   reactive policy at three translation qualities, per suite
  fig_het_forecast.pdf   reactive / forecast / gated, at the same three qualities

Formatting follows the simulator's own plotter (scheduling/src/plotter.py) so these read as part of
the same study. That means the temporal-group colour semantics (blue reactive, orange forecast, green
perfect future), the red dashed oracle reference at 1.0, black bar edges, bold value labels, and a
y-grid at alpha 0.3.

The hues are adjusted. plotter.py's own triple fails the palette checks, with #4e79a7 below the
chroma floor and #59a14f against #f28e2b at deltaE 3.8 under protanopia, which is indistinguishable.
These are the accessible versions of the same three roles, and they pass every check.

Colour means the same thing in both figures. It encodes the POLICY (blue reactive, orange forecast,
green gated), and translation quality is the x-axis in both, so a policy never changes colour
between the two.

Usage: plot_hetero_figures.py [out_dir]
"""
import os, re, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    '..', '..', '..'))
RES = os.path.join(REPO, 'results/scheduling/hetero')
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, 'figures')

NORM = 'Proactive_Hetero_Oracle'
REACT, FC, GATE = ('Model_Reactive_Hetero', 'Model_Forecast_Hetero',
                   'Model_Forecast_ReactiveGated_Hetero')
HEUR = ['EAS_Hetero', 'EAS_With_DVFS', 'Micro_EAS', 'Thread_Director',
        'Threshold_Migration', 'UCB1_Hetero']
EXCLUDE = ('spec_772.marian_r', 'spec_706.stockfish_r')

# One role-to-colour map, shared with per_workload_bars.py so a policy keeps its colour
# across every figure in the chapter:
#   heuristic #8a8a86 · reactive #0072B2 · forecast #D55E00 · gated #009E73
#   perfect future #E69F00 (used by the per-workload figure, not by these two)
# The four chromatic roles pass every palette check together, worst adjacent CVD separation
# deltaE 11.0 under deuteranopia.
C_HEUR = '#8a8a86'
C_REACT, C_FC, C_GATE = '#0072B2', '#D55E00', '#009E73'
C_PF = '#E69F00'
C_ORACLE = '#e41a1c'    # oracle reference, same red plotter.py uses

# Match the tables and report_chapter5.py: the three cross-processor model qualities are
# loocv/general/perfectcp. (The 'oracle' dir is the full Viterbi oracle, NOT perfect-CP; the
# 'gentemporal' dir is a different setup than the deployable 'general' arm.)
STUDIES = [('loocv', 'LOOCV'), ('general', 'General'), ('perfectcp', 'Perfect-CP')]
# Panel titles use the same spelling as the house figures (common.py SUITE_LABEL).
SUITE_LABEL = {'SPEC2017': 'SPEC 2017', 'SPEC2026': 'SPEC 2026', 'DaCapo': 'DaCapo'}


def suite(w):
    if w.startswith('dacapo'):
        return 'DaCapo'
    m = re.match(r'spec_(\d+)', w)
    return 'SPEC2017' if m and int(m.group(1)) < 700 else 'SPEC2026'


def load(study, metric):
    e = pd.read_csv(os.path.join(RES, study, 'all_phases_summary.csv'))
    e['S'] = e.Workload.map(suite)
    e = e[(~e.Workload.isin(EXCLUDE)) & (e.Metric == metric)]
    return e.pivot_table(index=['Workload', 'Phase', 'S'], columns='Policy',
                         values='Final_Value')


def ratio(pv, pol, suite_name):
    if pol not in pv.columns:
        return np.nan
    m = pv.index.get_level_values('S') == suite_name
    return float((pv[pol][m] / pv[NORM][m]).mean())


def finish(ax, oracle_label=None):
    ax.axhline(1.0, color=C_ORACLE, ls='--', lw=2, zorder=4, label=oracle_label)
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)


def draw(ax, x, vals, w, color, label):
    # No per-bar value labels. The exact numbers live in the prose and the ladder table,
    # and the Okabe-Ito hues clear the 3:1 contrast floor, so labels are not needed as
    # relief either.
    ax.bar(x, vals, w, color=color, label=label, zorder=3,
           edgecolor='black', linewidth=0.8)


def save(fig, name):
    fig.tight_layout()
    p = os.path.join(OUT, f'{name}.pdf')
    fig.savefig(p, bbox_inches='tight')
    fig.savefig(p.replace('.pdf', '.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {p}')


def _legend_below(fig, ax):
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=len(l), frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, -0.04))


def fig_reactive(metric='EDP'):
    # Line sweep over translation quality, one panel per suite, styled like the DVFS model
    # sweeps (fig04). The reactive model against the best heuristic and the oracle.
    suites = ['SPEC2017', 'SPEC2026', 'DaCapo']
    data = {s: load(s, metric) for s, _ in STUDIES}
    fig, axes = plt.subplots(1, len(suites), figsize=(2.6 * len(suites), 2.6),
                             sharey=True, squeeze=False)
    xs = list(range(len(STUDIES)))
    for c, sname in enumerate(suites):
        ax = axes[0][c]
        vals = [ratio(data[st], REACT, sname) for st, _ in STUDIES]
        ax.plot(xs, vals, '-o', color=C_REACT, lw=1.6, ms=5,
                label='Reactive' if c == 0 else None)
        pv = data['general']
        best = np.nanmin([ratio(pv, h, sname) for h in HEUR if h in pv.columns])
        ax.axhline(best, color=C_HEUR, ls='--', lw=1.4, zorder=2,
                   label='Best heuristic' if c == 0 else None)
        ax.axhline(1.0, color=C_ORACLE, ls='--', lw=1.6, zorder=2,
                   label='Oracle = 1.0' if c == 0 else None)
        ax.set_xticks(xs)
        ax.set_xticklabels([l for _, l in STUDIES], fontsize=8)
        ax.set_title(SUITE_LABEL[sname], fontsize=9)
        ax.grid(axis='y', alpha=0.3); ax.set_axisbelow(True)
    axes[0][0].set_ylabel('Norm. EDP', fontsize=9)
    axes[0][0].set_ylim(0.95, 1.62)
    fig.tight_layout()
    _legend_below(fig, axes[0][0])
    save(fig, 'fig_het_reactive')


def fig_forecast(metric='EDP'):
    # Line sweep over translation quality, one panel per suite, styled like the DVFS model
    # sweeps. Not sharey, since DaCapo's forecast reaches 1.80 against roughly 1.20 on SPEC.
    suites = ['SPEC2017', 'SPEC2026', 'DaCapo']
    data = {s: load(s, metric) for s, _ in STUDIES}
    fig, axes = plt.subplots(1, len(suites), figsize=(2.9 * len(suites), 2.6),
                             sharey=False, squeeze=False)
    series = ((REACT, 'Reactive', C_REACT, '-o'), (FC, 'Forecast', C_FC, '-^'),
              (GATE, 'Gated forecast', C_GATE, '-s'))
    xs = list(range(len(STUDIES)))
    for c, sname in enumerate(suites):
        ax = axes[0][c]
        top = 1.0
        for pol, lab, col, style in series:
            vals = [ratio(data[st], pol, sname) for st, _ in STUDIES]
            top = max(top, np.nanmax(vals))
            ax.plot(xs, vals, style, color=col, lw=1.6, ms=5, label=lab if c == 0 else None)
        ax.axhline(1.0, color=C_ORACLE, ls='--', lw=1.6, zorder=2,
                   label='Oracle = 1.0' if c == 0 else None)
        ax.set_xticks(xs)
        ax.set_xticklabels([l for _, l in STUDIES], fontsize=8)
        ax.set_title(SUITE_LABEL[sname], fontsize=9)
        ax.set_ylim(0.95, 1.0 + (top - 1.0) * 1.3)
        ax.grid(axis='y', alpha=0.3); ax.set_axisbelow(True)
    axes[0][0].set_ylabel('Norm. EDP', fontsize=9)
    fig.tight_layout()
    _legend_below(fig, axes[0][0])
    save(fig, 'fig_het_forecast')


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    fig_reactive()
    fig_forecast()

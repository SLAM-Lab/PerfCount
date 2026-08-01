#!/usr/bin/env python3
# Per-workload heterogeneous comparison, one figure per SPEC suite. At each x-axis point a
# workload, with four bars:
#   Reactive Heuristic        - best single reactive heuristic (suite-wide winner, not a
#                               per-workload min, which would be an oracle over heuristics)
#   Reactive General          - Model_Reactive_Hetero            (general/gentemporal study)
#   Gated Forecast General    - Model_Forecast_ReactiveGated_Hetero (general study)
#   Perfect Future+Perfect-CP - Model_Greedy_Oracle_Hetero       (oracle/perfect-CP study)
# All normalized to the full-trace Viterbi global oracle. Workloads are sorted by the gated
# forecast's gain over the reactive general model, so per-workload wins that the suite mean
# masks appear on the left.
#
# Usage:  RES=<results/scheduling> python per_workload_bars.py [out_dir] [metric]
import os, sys, re
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

RES = os.environ.get('RES', 'results/scheduling')
OUT = sys.argv[1] if len(sys.argv) > 1 else f'{RES}/hetero/per_workload'
METRIC = sys.argv[2] if len(sys.argv) > 2 else 'EDP'
# Which deployable translation model fills the two middle bars. The heuristic bar is
# model-independent and the perfect-future bar always comes from the perfect-CP study.
STUDY = sys.argv[3] if len(sys.argv) > 3 else 'general'
SLABEL = {'general': 'General', 'gentemporal': 'General', 'loocv': 'LOOCV',
          'geninsample': 'Gen-insample'}.get(STUDY, STUDY)
# The canonical no-suffix filename (what the chapter includes) is produced by the canonical
# 'general' arm; other studies get a suffixed filename so they cannot overwrite it.
NOSUFFIX = STUDY in ('general', 'gentemporal')
os.makedirs(OUT, exist_ok=True)

NORM = 'Proactive_Hetero_Oracle'
REACT = 'Model_Reactive_Hetero'
GATE = 'Model_Forecast_ReactiveGated_Hetero'
PF = 'Model_Greedy_Oracle_Hetero'
HEUR = ['EAS_Hetero', 'EAS_With_DVFS', 'Micro_EAS',
        'Thread_Director', 'Threshold_Migration', 'UCB1_Hetero']

# Excluded from the per-workload breakdown. These two are ~900x shorter than every other
# SPEC2026 workload (marian 90 chunks / 2 phases, stockfish 117 / 3, against 81k+ for the
# rest), so there is not enough history to forecast from and a single migration's warmup is
# a large fraction of the trace. They are outside the target use case rather than a result.
EXCLUDE = {'spec_772.marian_r', 'spec_706.stockfish_r'}


def suite_of(w):
    if w.startswith('dacapo'):
        return 'DaCapo'
    m = re.match(r'spec_(\d+)', w)
    if not m:
        return 'other'
    return 'SPEC2017' if int(m.group(1)) < 700 else 'SPEC2026'


def load(study, suite):
    E = pd.read_csv(f'{RES}/hetero/{study}/all_phases_summary.csv')
    E = E[(E.Metric == METRIC) & (E.Workload.map(suite_of) == suite)]
    E = E[~E.Workload.isin(EXCLUDE)]
    return E.pivot_table(index=['Workload', 'Phase'], columns='Policy', values='Final_Value')



def _heterogeneity(suite):
    """Fraction of chunks off the workload's modal optimal configuration, from the oracle.

    A property of the workload rather than of any policy, so it can be related to policy gains
    without circularity. Post-hoc only: a scheduler cannot observe it at runtime.
    """
    import ast, collections
    f = f'{RES}/hetero/{STUDY}/diagnostics.csv'
    if not os.path.exists(f):
        return pd.Series(dtype=float)
    d = pd.read_csv(f)
    d = d[(d.Metric == METRIC) & (d.Policy == NORM) & (d.Workload.map(suite_of) == suite)]
    out = {}
    for wl, g in d.groupby('Workload'):
        agg = collections.Counter(); tot = 0.0
        for _, r in g.iterrows():
            try:
                cf = ast.literal_eval(r.config_fracs)
            except Exception:
                continue
            for c, fr in cf.items():
                agg[c] = agg.get(c, 0.0) + float(fr) * float(r.n_chunks)
            tot += float(r.n_chunks)
        if tot > 0:
            out[wl] = 1.0 - max(agg.values()) / tot
    return pd.Series(out, dtype=float)


def per_wl(P, pol):
    """Per-workload mean of policy/oracle, averaged over that workload's phases."""
    return (P[pol] / P[NORM]).groupby('Workload').mean()


COLORS = ['#8a8a86', '#0072B2', '#009E73', '#E69F00']
SUITE_LABEL = {'SPEC2017': 'SPEC 2017', 'SPEC2026': 'SPEC 2026', 'DaCapo': 'DaCapo'}


def prepare(suite):
    """Per-workload table for one suite, sorted and split. None if the suite is absent."""
    gen = load(STUDY, suite)
    # Perfect-future comes from the SAME study, not a separate 'oracle' run. That policy scores
    # with true per-sample times, so its value does not depend on which prediction set was
    # loaded, and taking it from a different run risks mixing conventions -- the old 'oracle'
    # study pinned the cross-frequency half to top4 and predates the gate warmup fix, so its
    # bars were not comparable to the other three.
    orc = gen
    if gen.empty:
        return None
    hmeans = {h: (gen[h] / gen[NORM]).mean() for h in HEUR if h in gen.columns}
    best_h = min(hmeans, key=hmeans.get)
    series = {
        f'Reactive Heuristic ({best_h.replace("_Hetero", "")})': per_wl(gen, best_h),
        f'Reactive {SLABEL}': per_wl(gen, REACT),
        f'Gated Forecast {SLABEL}': per_wl(gen, GATE),
        'Perfect Future + Perfect-CP': per_wl(orc, PF),
    }
    df = pd.DataFrame(series).dropna()
    if df.empty:
        return None
    # Heterogeneity measured on the DECISION, from the oracle's own choices, not on counter
    # clustering. Counting phase files counts surviving GMM clusters at a fixed requested k,
    # which is a pipeline artifact, and refitting with BIC saturates at large n.
    het = _heterogeneity(suite)
    thr = het.median() if len(het) else 0.0
    multi = [w for w in df.index if het.get(w, 0.0) > thr]
    mono = [w for w in df.index if het.get(w, 0.0) <= thr]
    rcol, gcol = f'Reactive {SLABEL}', f'Gated Forecast {SLABEL}'

    def bygain(ws):
        sub = df.loc[ws]
        return list((sub[rcol] - sub[gcol]).sort_values(ascending=False).index)

    multi, mono = bygain(multi), bygain(mono)
    df = df.loc[multi + mono]
    summ = pd.DataFrame({'HETEROGENEOUS MEAN': df.loc[multi].mean(),
                         'HOMOGENEOUS MEAN': df.loc[mono].mean(),
                         'OVERALL MEAN': df.mean()}).T
    labels = ([w.replace('spec_', '').replace('dacapo_', '').replace('_r', '')
               for w in df.index]
              + ['', 'HETEROG.\nMEAN', 'HOMOG.\nMEAN', 'OVERALL\nMEAN'])
    heights = np.vstack([df.values, np.full((1, df.shape[1]), np.nan), summ.values])
    return dict(df=df, summ=summ, labels=labels, heights=heights, multi=multi,
                mono=mono, best_h=best_h, cols=list(df.columns), rcol=rcol, gcol=gcol)


def draw_panel(ax, d, suite, legend=False, xlabels=True, title=True):
    labels, heights, df = d['labels'], d['heights'], d['df']
    n = len(labels)
    x = np.arange(n); w = 0.2
    for j, col in enumerate(d['cols']):
        ax.bar(x + (j - 1.5) * w, heights[:, j], w, label=col if legend else None,
               color=COLORS[j], edgecolor='black', linewidth=0.3, zorder=3)
    ax.axhline(1.0, color='#e41a1c', ls='--', lw=1.5, zorder=4,
               label='Global oracle = 1.0' if legend else None)
    # Headroom above the tallest bar. A percentile-based ceiling clipped the tall
    # heuristic bars, which silently understates the baseline being beaten.
    top = np.nanmax(heights) * 1.10
    ax.set_ylim(0.95, top)
    for xv in (len(d['multi']) - 0.5, len(df) - 0.5):
        ax.axvline(xv, color='0.35', ls=':', lw=1.4, zorder=2)
    ax.text((len(d['multi']) - 1) / 2, top, f'heterogeneous (n={len(d["multi"])})',
            ha='center', va='top', fontsize=8, style='italic', color='0.3')
    ax.text(len(d['multi']) + (len(d['mono']) - 1) / 2, top,
            f'homogeneous (n={len(d["mono"])})',
            ha='center', va='top', fontsize=8, style='italic', color='0.3')
    ax.set_xticks(x)
    if xlabels:
        ax.set_xticklabels(labels, rotation=60, ha='right', fontsize=7)
        for t, lab in zip(ax.get_xticklabels(), labels):
            if 'MEAN' in lab:
                t.set_fontweight('bold')
    else:
        ax.set_xticklabels([])
    ax.set_xlim(-0.7, n - 0.3)
    ax.set_ylabel(f'Norm. {METRIC}', fontsize=9)
    if title:
        ax.set_title(SUITE_LABEL.get(suite, suite), fontsize=10, loc='left')
    ax.grid(axis='y', alpha=0.25, lw=0.5)
    ax.set_axisbelow(True)


SUITES = ['SPEC2017', 'SPEC2026', 'DaCapo']
prepared = {}
for suite in SUITES:
    d = prepare(suite)
    if d is None:
        continue
    prepared[suite] = d
    n = len(d['labels'])
    fig, ax = plt.subplots(figsize=(max(12, n * 0.62), 7))
    draw_panel(ax, d, suite, legend=True)
    ax.legend(fontsize=9, loc='upper left')
    fig.tight_layout()
    tag = '' if NOSUFFIX else f'_{SLABEL}'
    f = f'{OUT}/Hetero_{suite}_{METRIC}{tag}_per_workload.png'
    fig.savefig(f, dpi=130); plt.close(fig)
    print(f'wrote {f}  ({len(d["df"])} workloads: {len(d["multi"])} multi / '
          f'{len(d["mono"])} mono, best heuristic = {d["best_h"]})')
    g = lambda r: (r[d['rcol']] - r[d['gcol']]) / r[d['rcol']] * 100
    print(f'    gate-vs-reactive: heterog {g(d["summ"].loc["HETEROGENEOUS MEAN"]):+.2f}%  '
          f'homog {g(d["summ"].loc["HOMOGENEOUS MEAN"]):+.2f}%')

# Per-suite PDFs for the chapter, one per LaTeX subfigure. Sized for a \columnwidth slot
# in a stacked figure[p], so they are wider and shorter than the standalone PNGs. The legend
# rides on the first suite only, since the three share one encoding and sit on one page.
pdf_dir = os.environ.get('PDF_OUT')
if pdf_dir and prepared:
    os.makedirs(pdf_dir, exist_ok=True)
    tag = '' if NOSUFFIX else f'_{SLABEL}'
    ss = [s_ for s_ in SUITES if s_ in prepared]
    for i, s_ in enumerate(ss):
        d = prepared[s_]
        n = len(d['labels'])
        fig, ax = plt.subplots(figsize=(max(9.5, n * 0.40), 3.0))
        # No in-axes title: the LaTeX subcaption names the suite.
        draw_panel(ax, d, s_, legend=(i == 0), title=False)
        if i == 0:
            ax.legend(fontsize=7.5, ncol=5, loc='upper center',
                      bbox_to_anchor=(0.5, 1.30), frameon=False)
        fig.tight_layout()
        out = f'{pdf_dir}/fig_het_perworkload_{s_}_{METRIC}{tag}.pdf'
        fig.savefig(out, bbox_inches='tight')
        plt.close(fig)
        print(f'wrote {out}')

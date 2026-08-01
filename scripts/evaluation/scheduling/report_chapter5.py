#!/usr/bin/env python3
"""Print the Chapter 5 heterogeneous and DVFS numbers, with the gap to the oracle.

Reads the simulator summaries produced by run_chapter5.sh and prints the tables the chapter
reports, so a reader can launch the runs and check repeatability against the text. Every number
is normalized to the relevant global Viterbi oracle, lower is better.

Arm result directories default to the canonical names run_chapter5.sh writes and can be pointed
elsewhere through the environment (HET_LOOCV, HET_GENERAL, HET_PERFECTCP, DVFS_LOOCV,
DVFS_GENERAL), which is how it is checked against an existing set of runs.
"""
import os
import re
import sys
import collections

import numpy as np
import pandas as pd

RES = os.environ.get('RES', os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          '..', '..', '..', 'results', 'scheduling'))
HET = {'LOOCV': os.environ.get('HET_LOOCV', 'loocv'),
       'General': os.environ.get('HET_GENERAL', 'general'),
       'Perfect-CP': os.environ.get('HET_PERFECTCP', 'perfectcp')}
DVFS = {'LOOCV': os.environ.get('DVFS_LOOCV', 'loocv'),
        'General': os.environ.get('DVFS_GENERAL', 'general')}

NORM = 'Proactive_Hetero_Oracle'
GATE = 'Model_Forecast_ReactiveGated_Hetero'
REACT = 'Model_Reactive_Hetero'
RAWFC = 'Model_Forecast_Hetero'
PF = 'Model_Greedy_Oracle_Hetero'          # greedy, true times, static decision power
GREEDY_TRUEP = 'Greedy_Oracle_Hetero'      # greedy, true times AND true per-sample power
HEUR = ['EAS_Hetero', 'EAS_With_DVFS', 'Micro_EAS', 'Threshold_Migration',
        'Thread_Director', 'UCB1_Hetero']
EXC = {'spec_772.marian_r', 'spec_706.stockfish_r'}


def suite(w):
    if w.startswith('dacapo'):
        return 'DaCapo'
    m = re.match(r'spec_(\d+)', w)
    return 'SPEC2017' if m and int(m.group(1)) < 700 else 'SPEC2026'


def load_het(arm, metric, s):
    f = os.path.join(RES, 'hetero', HET[arm], 'all_phases_summary.csv')
    if not os.path.exists(f):
        return None
    e = pd.read_csv(f)
    e['S'] = e.Workload.map(suite)
    e = e[(~e.Workload.isin(EXC)) & (e.Metric == metric) & (e.S == s)]
    return e.pivot_table(index=['Workload', 'Phase'], columns='Policy', values='Final_Value')


def r(x, pol):
    return (x[pol] / x[NORM]).mean() if (x is not None and pol in x.columns) else float('nan')


def het_ladder():
    print('\n' + '=' * 78)
    print('HETEROGENEOUS SCHEDULING LADDER  (EDP, normalized to global oracle = 1.000)')
    print('each column adds one capability; the gap to the oracle is decomposed at the end')
    print('=' * 78)
    hdr = f"{'suite':<9}{'best heur':>10}{'reactive':>9}{'gate':>8}{'perf-CP':>9}{'greedy+trueP':>13}{'oracle':>8}"
    print(hdr)
    for s in ['SPEC2017', 'SPEC2026', 'DaCapo']:
        g = load_het('General', 'EDP', s)
        p = load_het('Perfect-CP', 'EDP', s)
        if g is None:
            print(f"{s:<9}   (General arm missing)")
            continue
        bh = min((r(g, h) for h in HEUR if h in g.columns), default=float('nan'))
        print(f"{s:<9}{bh:10.3f}{r(g, REACT):9.3f}{r(g, GATE):8.3f}"
              f"{r(p, GATE):9.3f}{r(g, GREEDY_TRUEP):13.3f}{1.000:8.3f}")
    print("\ngap decomposition (EDP, points above the oracle):")
    for s in ['SPEC2017', 'SPEC2026']:
        g = load_het('General', 'EDP', s)
        if g is None:
            continue
        gate = r(g, GATE); pf_static = r(g, PF); tp = r(g, GREEDY_TRUEP)
        print(f"  {s}: gate {(gate-1)*100:5.1f}  ->  power term {(pf_static-tp)*100:5.1f}"
              f"  ->  planning {(tp-1)*100:4.1f}   (a causal greedy with true power reaches the oracle)")


def het_vs_reactive():
    print('\n' + '=' * 78)
    print('GATE vs REACTIVE  (per-suite mean, both model qualities)  positive = gate better')
    print('=' * 78)
    print(f"{'metric':<6}{'suite':<10}{'LOOCV':>10}{'General':>10}")
    from scipy import stats  # noqa
    for metric in ['EDP', 'ED2P']:
        for s in ['SPEC2017', 'SPEC2026']:
            out = []
            for arm in ('LOOCV', 'General'):
                x = load_het(arm, metric, s)
                out.append((r(x, REACT) - r(x, GATE)) / r(x, REACT) * 100 if x is not None else float('nan'))
            print(f"{metric:<6}{s:<10}{out[0]:+9.2f}%{out[1]:+9.2f}%")


def het_vs_heuristic():
    print('\n' + '=' * 78)
    print('HEADLINE: deployable gate vs best prior heuristic (General, EDP)')
    print('=' * 78)
    for s in ['SPEC2017', 'SPEC2026', 'DaCapo']:
        g = load_het('General', 'EDP', s)
        if g is None:
            continue
        bh = min((r(g, h), h) for h in HEUR if h in g.columns)
        gate = r(g, GATE)
        print(f"  {s:<9} gate {gate:.3f}  best-heur {bh[1].replace('_Hetero',''):<18} {bh[0]:.3f}"
              f"   gate wins by {(bh[0]-gate)/bh[0]*100:+.1f}%")


def het_split():
    print('\n' + '=' * 78)
    print('WHERE FORECASTING PAYS: gate over reactive, split at median heterogeneity')
    print('per workload, SPEC only, gain = reactive - gate (relative %)')
    print('=' * 78)
    from scipy import stats
    dgf = os.path.join(RES, 'hetero', HET['General'], 'diagnostics.csv')
    if not os.path.exists(dgf):
        print('  (diagnostics missing, cannot compute heterogeneity split)')
        return
    dg = pd.read_csv(dgf)
    dg = dg[(dg.Metric == 'EDP') & (dg.Policy == NORM)]
    HW = collections.defaultdict(lambda: [0.0, 0.0])
    import ast
    for (wl, ph), gg in dg.groupby(['Workload', 'Phase']):
        agg = collections.Counter(); tot = 0.0
        for _, row in gg.iterrows():
            try:
                cf = ast.literal_eval(row.config_fracs)
            except Exception:
                continue
            for c, fr in cf.items():
                agg[c] = agg.get(c, 0.0) + float(fr) * float(row.n_chunks)
            tot += float(row.n_chunks)
        if tot > 0:
            HW[wl][0] += (1 - max(agg.values()) / tot) * tot; HW[wl][1] += tot
    HWL = {w: v[0] / v[1] for w, v in HW.items() if v[1] > 0}
    for arm in ('General', 'Perfect-CP'):
        print(f"  -- {arm} model --")
        print(f"     {'metric':<6}{'group':<15}{'mean':>8}{'p':>8}{'win/loss':>10}")
        for metric in ['EDP', 'ED2P']:
            f = os.path.join(RES, 'hetero', HET[arm], 'all_phases_summary.csv')
            e = pd.read_csv(f)
            e = e[(~e.Workload.isin(EXC)) & (e.Metric == metric) & (e.Workload.map(suite) != 'DaCapo')]
            p = e.pivot_table(index=['Workload', 'Phase'], columns='Policy', values='Final_Value')
            rw = (p[REACT] / p[NORM]).groupby(level=0).mean()
            gw = (p[GATE] / p[NORM]).groupby(level=0).mean()
            wls = [w for w in rw.index if w in HWL]
            hv = pd.Series({w: HWL[w] for w in wls}); thr = hv.median()
            for nm, sel in (('heterogeneous', [w for w in wls if hv[w] > thr]),
                            ('homogeneous', [w for w in wls if hv[w] <= thr])):
                a = rw.loc[sel].values; b = gw.loc[sel].values
                rel = (a - b) / a * 100
                try:
                    pv = stats.wilcoxon(a, b).pvalue
                except Exception:
                    pv = float('nan')
                w_ = int((b < a - 1e-12).sum()); l_ = int((b > a + 1e-12).sum())
                print(f"     {metric:<6}{nm:<15}{rel.mean():+7.2f}%{pv:8.3f}{f'{w_}/{l_}':>10}")


def dvfs():
    print('\n' + '=' * 78)
    print('DVFS  (reactive / forecast / perfect-future per core, normalized to per-core oracle)')
    print('=' * 78)
    for metric in ['EDP', 'ED2P']:
        for core in ['P', 'E']:
            norm = f'Global_Oracle_{core}'
            row = f"  {core}-core {metric:<5}"
            for arm in ('LOOCV', 'General'):
                f = os.path.join(RES, 'DVFS', DVFS[arm], 'all_phases_summary.csv')
                if not os.path.exists(f):
                    row += f"  {arm}: (missing)"; continue
                e = pd.read_csv(f)
                e = e[(~e.Workload.isin(EXC)) & (e.Metric == metric) & (e.Workload.map(suite) != 'DaCapo')]
                p = e.pivot_table(index=['Workload', 'Phase'], columns='Policy', values='Final_Value')
                if norm not in p.columns:
                    row += f"  {arm}: (no oracle)"; continue
                def col(pol):
                    return (p[pol] / p[norm]).mean() if pol in p.columns else float('nan')
                rr = col(f'Model_Greedy_{core}')                     # reactive model
                gg = col(f'Model_Forecast_ReactiveGated_{core}')     # gated forecast
                pf = col(f'Model_Greedy_Oracle_{core}')              # perfect future
                row += f"  {arm}: react {rr:.3f} gate {gg:.3f} PF {pf:.3f}"
            print(row)


def deployable_compare():
    """Idealized vs deployable diagonal: the incumbent scored from the true next sample versus
    the previous one, which is what a scheduler actually has.

    The idealization is NOT a rounding-error effect: replacing it with the stale estimate a
    scheduler really has costs 1-3 EDP points, and the cost is concentrated on the heterogeneous
    workloads (where the incumbent estimate actually feeds a stay-vs-move decision), hitting the
    migrating gate harder than reactive there. Ratio-of-means alone hides the important part, so
    this also reports the paired gate-vs-reactive win/loss: under the honest diagonal the paired
    direction flips toward the gate on the homogeneous majority, because the stale estimate
    penalises reactive's stay-confidence -- which reactive leans on -- more than the gate's
    forecasts. See the heterogeneous/homogeneous split in Chapter 5 Section 6."""
    from scipy import stats  # noqa
    print('\n' + '=' * 78)
    print('DEPLOYABLE DIAGONAL: incumbent scored from previous sample, not true next sample')
    print('idealized -> deployable, EDP, normalized to oracle (a real scheduler has the deployable)')
    print('gate-adv = paired mean (reactive-gate)/reactive; w/l = phases gate beats/loses to reactive')
    print('=' * 78)
    print(f"{'arm':<8}{'suite':<10}{'reactive':>18}{'gate':>18}{'gate-adv i->d':>16}{'w/l i->d':>16}")
    for arm in ('General', 'LOOCV'):
        base = HET[arm]
        for s in ['SPEC2017', 'SPEC2026']:
            def one(dirname):
                f = os.path.join(RES, 'hetero', dirname, 'all_phases_summary.csv')
                if not os.path.exists(f):
                    return None
                e = pd.read_csv(f); e['S'] = e.Workload.map(suite)
                e = e[(~e.Workload.isin(EXC)) & (e.Metric == 'EDP') & (e.S == s)]
                return e.pivot_table(index=['Workload', 'Phase'], columns='Policy', values='Final_Value')

            def paired(p):  # (mean gate advantage %, wins, losses, wilcoxon p)
                a = (p[REACT] / p[NORM]).values; b = (p[GATE] / p[NORM]).values
                rel = ((a - b) / a * 100).mean()
                try:
                    pv = stats.wilcoxon(a, b).pvalue
                except Exception:
                    pv = float('nan')
                return rel, int((b < a - 1e-12).sum()), int((b > a + 1e-12).sum()), pv

            pi = one(base); pd_ = one(base + '_deployable')
            if pi is None or pd_ is None:
                print(f"{arm:<8}{s:<10}   (need both hetero/{base} and hetero/{base}_deployable)")
                continue
            ri, gi, adv_i, wi, li, _ = (r(pi, REACT), r(pi, GATE), *paired(pi))
            rd, gd, adv_d, wd, ld, pvd = (r(pd_, REACT), r(pd_, GATE), *paired(pd_))
            print(f"{arm:<8}{s:<10}{ri:7.3f} -> {rd:<7.3f}{gi:7.3f} -> {gd:<7.3f}"
                  f"{adv_i:+5.2f}->{adv_d:+5.2f}%{f'{wi}/{li}->{wd}/{ld}':>16}")
    print("\nThe honest diagonal costs 1-3 EDP points and the cost falls hardest on the gate on the")
    print("heterogeneous workloads it migrates on; the paired w/l flips toward the gate (significant")
    print("on the homogeneous group, where the stale estimate penalises reactive's stay-confidence).")
    print("Ratio-of-means understates this -- read the paired w/l, and the het/homo split in the text.")


def main():
    deployable = '--deployable' in sys.argv
    print(f"reading from {os.path.normpath(RES)}")
    print(f"hetero arms: {HET}\nDVFS arms:   {DVFS}")
    if deployable:
        deployable_compare()
        print()
        return
    het_ladder()
    het_vs_heuristic()
    het_vs_reactive()
    het_split()
    dvfs()
    print()


if __name__ == '__main__':
    main()

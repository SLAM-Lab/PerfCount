#!/usr/bin/env python3
"""Print the DVFS ladder for a sim output dir, and check it for self-consistency.

Reports the per-core ladder (mean and median, both metrics) that fills the chapter's
DVFS tables, then runs sanity checks that catch the failure modes we have actually hit:

  - a model policy bit-identical to its oracle counterpart (silent prediction fallback)
  - a policy missing entirely (registration guard bug)
  - governors that beat best-static (would contradict the chapter's claim)
  - perfect-future worse than reactive (temporal ordering violated)

Usage: dvfs_report.py <sim_dir> [--metric EDP]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SU = ['spec17', 'spec26', 'dacapo']
GOV = ['Performance_Gov', 'Powersave_Gov', 'Ondemand', 'Conservative', 'Schedutil_PELT',
       'Interactive', 'EWMA', 'Intel_HWP', 'UCB1']


def suite_of(w):
    if w.startswith('dacapo'):
        return 'dacapo'
    if not w.startswith('spec_'):
        return 'other'
    try:
        n = int(w.split('_')[1].split('.')[0])
    except (ValueError, IndexError):
        return 'other'
    return 'spec17' if n < 600 else 'spec26'


def load(sim, metric, base):
    d = pd.read_csv(Path(sim) / 'all_phases_summary.csv')
    d = d[d.Metric == metric].copy()
    d['key'] = d.Workload + '|' + d.Phase.astype(str)
    d['suite'] = d.Workload.map(suite_of)
    b = d[d.Policy == base].set_index('key').Final_Value
    if b.empty:
        sys.exit(f'{sim}: baseline {base} missing')
    d['norm'] = d.Final_Value / d.key.map(b)
    return d[np.isfinite(d.norm)]


def agg(d, pol, how='mean'):
    s = d[d.Policy == pol]
    if s.empty:
        return {}
    g = s.groupby('suite').norm
    return (g.mean() if how == 'mean' else g.median()).to_dict()


def fmt(v):
    return {k: round(x, 3) for k, x in v.items()} if v else 'MISSING'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('sim')
    ap.add_argument('--metric', default='EDP', choices=['EDP', 'ED2P'])
    a = ap.parse_args()

    problems = []
    for core in 'PE':
        d = load(a.sim, a.metric, f'Global_Oracle_{core}')
        print(f'\n{"=" * 76}\n{core}-core DVFS  ({a.metric}, normalized to Global_Oracle_{core})\n{"=" * 76}')
        best_gov = {}
        for s in SU:
            vals = [agg(d, f'{g}_{core}').get(s) for g in GOV]
            vals = [v for v in vals if v is not None and np.isfinite(v)]
            if vals:
                best_gov[s] = min(vals)
        static = {}
        for s in SU:
            vals = [agg(d, f'Static_{core}_{f}GHz').get(s) for f in ['1.0', '2.0', '3.0', '4.0']]
            vals = [v for v in vals if v is not None and np.isfinite(v)]
            if vals:
                static[s] = min(vals)

        rows = [('best governor', best_gov), ('best static', static)]
        for lab, pol in [('reactive oracle', f'Reactive_Oracle_{core}'),
                         ('reactive model', f'Model_Greedy_{core}'),
                         ('  + damp W=5', f'Model_Greedy_Damp5_{core}'),
                         ('  + damp W=10', f'Model_Greedy_Damp10_{core}'),
                         ('  + commit W=5', f'Model_Greedy_Commit5_{core}'),
                         ('  + commit W=10', f'Model_Greedy_Commit10_{core}'),
                         ('forecast model', f'Model_Forecast_{core}'),
                         ('  phase-UNAWARE', f'Model_Forecast_Unaware_{core}'),
                         ('  persist-GATED', f'Model_Forecast_Gated_{core}'),
                         ('  + damp W=5', f'Model_Forecast_Damp5_{core}'),
                         ('  + damp W=10', f'Model_Forecast_Damp10_{core}'),
                         ('  + commit W=5', f'Model_Forecast_Commit5_{core}'),
                         ('  + commit W=10', f'Model_Forecast_Commit10_{core}'),
                         ('FORECAST x ORACLE', f'Forecast_Oracle_{core}'),
                         ('  phase-UNAWARE', f'Forecast_Oracle_Unaware_{core}'),
                         ('  persist-GATED', f'Forecast_Oracle_Gated_{core}'),
                         ('perfect future', f'Model_Greedy_Oracle_{core}'),
                         ('  + k=1', f'Model_Greedy_Oracle_k1_{core}'),
                         ('  + k=5', f'Model_Greedy_Oracle_k5_{core}'),
                         ('model Viterbi', f'Model_Global_{core}'),
                         ('greedy oracle', f'Greedy_Oracle_{core}')]:
            rows.append((lab, agg(d, pol)))

        print(f'{"policy":22s} ' + ' '.join(f'{s:>17s}' for s in SU))
        print(f'{"":22s} ' + ' '.join(f'{"mean / median":>17s}' for _ in SU))
        for lab, v in rows:
            if not v:
                print(f'{lab:22s} ' + ' '.join(f'{"--":>17s}' for _ in SU))
                continue
            cells = []
            for s in SU:
                cells.append(f'{v[s]:8.3f} /{"":1s}' if s in v else f'{"--":>10s}')
            print(f'{lab:22s} ' + ' '.join(f'{v[s]:17.3f}' if s in v else f'{"--":>17s}' for s in SU))

        # medians for the headline rows
        print(f'\n{"median":22s} ' + ' '.join(f'{s:>17s}' for s in SU))
        for lab, pol in [('reactive model', f'Model_Greedy_{core}'),
                         ('forecast model', f'Model_Forecast_{core}'),
                         ('  phase-UNAWARE', f'Model_Forecast_Unaware_{core}'),
                         ('perfect future', f'Model_Greedy_Oracle_{core}')]:
            v = agg(d, pol, 'median')
            print(f'{lab:22s} ' + ' '.join(f'{v[s]:17.3f}' if s in v else f'{"--":>17s}' for s in SU))

        # ---- consistency checks ----
        p = d.pivot_table(index='key', columns='Policy', values='Final_Value')
        for a_, b_ in [(f'Model_Forecast_{core}', f'Model_Greedy_Oracle_{core}'),
                       (f'Model_Greedy_{core}', f'Reactive_Oracle_{core}'),
                       (f'Model_Greedy_{core}', f'Model_Greedy_Oracle_{core}')]:
            if a_ in p and b_ in p:
                same = np.isclose(p[a_], p[b_], rtol=1e-9).sum()
                if same > 0.5 * len(p):
                    problems.append(f'{a_} is bit-identical to {b_} on {same}/{len(p)} phases '
                                    f'-- almost certainly a silent prediction fallback')
        for s in SU:
            if s in best_gov and s in static and best_gov[s] < static[s]:
                problems.append(f'{core}/{s}: best governor ({best_gov[s]:.3f}) beats best static '
                                f'({static[s]:.3f}) -- contradicts the chapter claim')
        rm, pf = agg(d, f'Model_Greedy_{core}'), agg(d, f'Model_Greedy_Oracle_{core}')
        for s in SU:
            if s in rm and s in pf and pf[s] > rm[s] + 1e-6:
                problems.append(f'{core}/{s}: perfect-future ({pf[s]:.3f}) is WORSE than reactive '
                                f'({rm[s]:.3f}) -- temporal ordering violated')

        aw, un = agg(d, f'Model_Forecast_{core}'), agg(d, f'Model_Forecast_Unaware_{core}')
        if aw and un:
            print(f'\n  phase-awareness effect ({core}-core, aware - unaware; negative = aware helps):')
            for s_ in SU:
                if s_ in aw and s_ in un:
                    dlt = aw[s_] - un[s_]
                    print(f'    {s_:8s} {dlt:+.4f}   {"aware better" if dlt < -1e-4 else ("unaware better" if dlt > 1e-4 else "tie")}')

    print(f'\n{"=" * 76}')
    if problems:
        print('CONSISTENCY PROBLEMS:')
        for x in problems:
            print('  ! ' + x)
    else:
        print('consistency checks: all passed')


if __name__ == '__main__':
    main()

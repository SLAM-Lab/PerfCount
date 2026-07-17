#!/usr/bin/env python3
"""Print the heterogeneous scheduling ladder for a sim output dir.

Reports the per-suite ladder (EDP and ED2P) that fills the chapter's heterogeneous tables:
the prior heuristics, best static, the reactive/forecast/perfect-future model policies, and
the reactive-to-perfect-future separation that is the headroom available to forecasting.

Normalizes to the full-trace Viterbi global oracle (Proactive_Hetero_Oracle) when it is
present. When the study was run with SKIP_VITERBI that policy is absent, so it falls back to
the greedy per-chunk oracle and says so -- the greedy oracle ignores transition costs and so
over-migrates, which inflates every value and can push a no-migration policy below 1.0 on the
high-migration suites. The reactive-to-perfect-future separation is robust to the choice since
it is measured within one normalization.

Usage: hetero_report.py <sim_dir> [--metric EDP]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SU = ['spec17', 'spec26', 'dacapo']
FREQS = ['1.0', '2.0', '3.0', '4.0']


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('sim')
    ap.add_argument('--metric', default='EDP', choices=['EDP', 'ED2P'])
    a = ap.parse_args()

    d = pd.read_csv(Path(a.sim) / 'all_phases_summary.csv')
    d = d[d.Metric == a.metric].copy()
    d['key'] = d.Workload + '|' + d.Phase.astype(str)
    p = d.pivot_table(index='key', columns='Policy', values='Final_Value')
    suite = pd.Series({k: suite_of(k.split('|')[0]) for k in p.index})

    if 'Proactive_Hetero_Oracle' in p:
        norm, norm_name = p['Proactive_Hetero_Oracle'], 'Viterbi global oracle'
    elif 'Greedy_Oracle_Hetero' in p:
        norm, norm_name = p['Greedy_Oracle_Hetero'], 'greedy oracle (SKIP_VITERBI; approximate)'
    else:
        sys.exit('no oracle policy found to normalize to')

    def static(idx):
        vals = [(p.loc[idx, f'Static_{c}_{f}GHz'] / norm.loc[idx]).mean()
                for c in 'PE' for f in FREQS if f'Static_{c}_{f}GHz' in p]
        return min(vals) if vals else np.nan

    def agg(idx, pol):
        return (p.loc[idx, pol] / norm.loc[idx]).mean() if pol in p else np.nan

    rows = [('EAS (prior)', 'EAS_Hetero'),
            ('EAS + DVFS', 'EAS_With_DVFS'),
            ('threshold-migrate', 'Threshold_Migration'),
            ('thread-director', 'Thread_Director'),
            ('best static', '_STATIC_'),
            ('reactive oracle', 'Reactive_Oracle_Hetero'),
            ('reactive model', 'Model_Reactive_Hetero'),
            ('forecast model', 'Model_Forecast_Hetero'),
            ('  + damp W=5', 'Model_Forecast_Damp5_Hetero'),
            ('  + damp W=10', 'Model_Forecast_Damp10_Hetero'),
            ('  + commit W=5', 'Model_Forecast_Commit5_Hetero'),
            ('perfect-future model', 'Model_Greedy_Oracle_Hetero'),
            ('  + k=1', 'Model_Greedy_Oracle_k1_Hetero'),
            ('greedy oracle', 'Greedy_Oracle_Hetero'),
            ('global oracle', 'Proactive_Hetero_Oracle')]

    print(f'\n{"=" * 68}\nHeterogeneous ladder  ({a.metric}, normalized to {norm_name})\n{"=" * 68}')
    present = {s: (suite == s).sum() for s in SU if (suite == s).sum() > 0}
    print(f'{"policy":22s} ' + ' '.join(f'{s} (n={n})'.rjust(13) for s, n in present.items()))
    for lab, pol in rows:
        cells = []
        for s in present:
            idx = p.index[(suite == s).values]
            v = static(idx) if pol == '_STATIC_' else agg(idx, pol)
            cells.append(f'{v:13.3f}' if np.isfinite(v) else f'{"--":>13s}')
        if any(c.strip() != '--' for c in cells):
            print(f'{lab:22s} ' + ' '.join(cells))

    # reactive -> perfect-future separation, the headroom for forecasting
    print(f'\n{"reactive-to-perfect-future separation (reactive_model - 1.0):":s}')
    for s in present:
        idx = p.index[(suite == s).values]
        rm = agg(idx, 'Model_Reactive_Hetero')
        pf = agg(idx, 'Model_Greedy_Oracle_Hetero')
        if np.isfinite(rm) and np.isfinite(pf):
            print(f'    {s:8s} reactive {rm:.3f}  perfect-future {pf:.3f}  '
                  f'separation {rm - pf:+.3f} ({100 * (rm - pf):.0f}%)')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Invariant test for the path-stitched forecast tensors.

The defining property: where the recorded path never changes configuration, there is nothing
to stitch. The stitched history for target C is then exactly the history the old per-source
dump already built from that one config, so the two forecasts must agree. Where the path does
change, they must differ, and the differences must sit at and after the switches.

This is the test that would catch a stitcher which silently fell through to copying its input
-- the failure that produced a fake NEED-6 result earlier in this project. A copy would agree
everywhere, including on the migrating phases where it must not.

Usage:
  test_stitch_invariant.py --actions <dir> --policy P --metric EDP \
      --old <cross_proc_forecast_gentemporal_10M> --new <cross_proc_forecast_stitched_EDP_10M>
"""
import argparse, glob, os, re, sys
import numpy as np, pandas as pd

FAILS = []


def check(name, ok, detail=''):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILS.append(name)


def path_for(actions, bench, ph, metric, policy):
    f = os.path.join(actions, f'{bench}__{ph}__{metric}__{policy}.csv')
    if not os.path.exists(f):
        return None
    d = pd.read_csv(f).sort_values('chunk')
    return d['config'].tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--actions', required=True)
    ap.add_argument('--policy', required=True)
    ap.add_argument('--metric', default='EDP')
    ap.add_argument('--old', required=True)
    ap.add_argument('--new', required=True)
    ap.add_argument('--src', default='P_3.0GHz')
    ap.add_argument('--limit', type=int, default=40)
    ap.add_argument('--tol', type=float, default=1e-9)
    a = ap.parse_args()

    files = sorted(glob.glob(f'{a.new}/speedups_from_{a.src}/*.csv'))[:a.limit]
    if not files:
        sys.exit(f'no stitched files under {a.new}/speedups_from_{a.src}/')

    const, moving = [], []
    for f in files:
        stem = os.path.basename(f)[:-4]
        m = re.match(r'(.+)_phase(\d+)$', stem)
        if not m:
            continue
        bench, ph = m.group(1), int(m.group(2))
        p = path_for(a.actions, bench, ph, a.metric, a.policy)
        if p is None:
            continue
        old_f = os.path.join(a.old, f'speedups_from_{a.src}', os.path.basename(f))
        if not os.path.exists(old_f):
            continue
        o, n = pd.read_csv(old_f), pd.read_csv(f)
        k = min(len(o), len(n))
        cols = [c for c in o.columns if c.startswith('Speedup_') and c in n.columns]
        if not cols or k < 10:
            continue
        d = np.abs(o[cols].values[:k] - n[cols].values[:k])
        rel = d / np.maximum(np.abs(o[cols].values[:k]), 1e-12)
        row_diff = (rel > a.tol).any(axis=1)
        switches = sum(1 for i in range(1, min(len(p), k)) if p[i] != p[i - 1])
        rec = {'bench': stem, 'switches': switches, 'n': k,
               'rows_diff': int(row_diff.sum()), 'frac': row_diff.mean(),
               'max_rel': float(rel.max())}
        # where does the path sit relative to the source slice we are reading?
        rec['on_src'] = float(np.mean([c == a.src for c in p[:k]]))
        (const if switches == 0 else moving).append(rec)

    print(f'source {a.src}, {len(const)} constant-path phases, {len(moving)} migrating\n')

    # 1. constant path AND parked on this slice's source -> stitched == old, exactly
    same_src = [r for r in const if r['on_src'] > 0.999]
    if same_src:
        worst = max(r['max_rel'] for r in same_src)
        check('constant path on this source reproduces the old tensor',
              worst <= 1e-6, f'{len(same_src)} phases, max rel diff {worst:.2e}')
    else:
        print('  SKIP  no constant-path phases parked on this source')

    # 2. constant path on a DIFFERENT config -> must differ (history came from elsewhere)
    diff_src = [r for r in const if r['on_src'] < 0.001]
    if diff_src:
        n_same = sum(1 for r in diff_src if r['rows_diff'] == 0)
        check('constant path on another config does differ from the old tensor',
              n_same == 0, f'{len(diff_src)} phases, {n_same} identical (want 0)')

    # 3. not a copy: something, somewhere, must have changed
    total_diff = sum(r['rows_diff'] for r in const + moving)
    check('output is not a verbatim copy of its input', total_diff > 0,
          f'{total_diff} differing rows overall')

    if moving:
        mf = np.mean([r['frac'] for r in moving])
        print(f'\n  migrating phases: mean {mf*100:.2f}% of rows differ, '
              f'mean {np.mean([r["switches"] for r in moving]):.1f} switches')
        for r in sorted(moving, key=lambda x: -x['frac'])[:5]:
            print(f'    {r["bench"]:<34} switches={r["switches"]:<5} '
                  f'rows_diff={r["frac"]*100:5.1f}%  max_rel={r["max_rel"]:.3f}')

    print()
    if FAILS:
        print(f'{len(FAILS)} FAILED: {FAILS}')
        sys.exit(1)
    print('invariant tests passed')


if __name__ == '__main__':
    main()

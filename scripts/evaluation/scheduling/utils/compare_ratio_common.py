#!/usr/bin/env python3
"""Compare prediction sets in the ratio domain on the COMMON set of workload-phases.

Prediction sets can cover different phases: a feature set is only usable where every one of its
counters was collected, and counter availability varies by workload as well as by core type. A
mean taken over each set's own coverage is therefore not a like-for-like comparison. This scores
every set on the intersection.

Usage: compare_ratio_common.py <src> <name>=<dir> [<name>=<dir> ...]
"""
import sys, os, glob
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xproc_eval import GRAN, EXCLUDE, MIN_SAMPLES


def phases_of(d, src):
    return set(os.path.basename(x)[:-4]
               for x in glob.glob(f'{d}/speedups_from_{src}/spec_*.csv')
               if not os.path.basename(x)[:-4].startswith(EXCLUDE))


def score(d, src, keep, deadband=0.02):
    rows = []
    for wl in sorted(keep):
        f = f'{d}/speedups_from_{src}/{wl}.csv'
        g = os.path.join(GRAN, f'speedups_{src}_{wl}.csv')
        if not (os.path.exists(f) and os.path.exists(g)):
            continue
        t = pd.read_csv(g)
        if len(t) < MIN_SAMPLES:
            continue
        p = pd.read_csv(f)
        n = min(len(t), len(p))
        for col in [c for c in p.columns if c.startswith('Speedup_')]:
            if col not in t.columns:
                continue
            tv = t[col].values[:n].astype(float)
            pv = p[col].values[:n].astype(float)
            m = np.isfinite(tv) & np.isfinite(pv) & (tv > 1e-6)
            if m.sum() < 50 or pv[m].std() < 1e-12:
                continue
            tv, pv = tv[m], pv[m]
            dec = np.abs(tv - 1.0) > deadband
            rows.append({'wl': wl, 'key': f'{wl}|{col}',
                         'corr': np.corrcoef(pv, tv)[0, 1],
                         'range': pv.std() / max(tv.std(), 1e-12),
                         'sign': (np.sign(pv[dec] - 1) == np.sign(tv[dec] - 1)).mean()
                                 if dec.sum() else np.nan,
                         'mape': np.mean(np.abs(pv - tv) / tv) * 100})
    return pd.DataFrame(rows)


def main():
    src = sys.argv[1]
    sets = dict(a.split('=', 1) for a in sys.argv[2:])
    cov = {k: phases_of(d, src) for k, d in sets.items()}
    common = set.intersection(*cov.values())
    print(f'source {src}\n')
    for k in sets:
        print(f'  {k:<24} covers {len(cov[k]):3d} phases')
    print(f'  {"COMMON":<24}        {len(common):3d}\n')
    ref = max(cov, key=lambda k: len(cov[k]))
    for k in sets:
        gap = cov[ref] - cov[k]
        if gap:
            print(f'  in {ref} but not {k} ({len(gap)}): {sorted(gap)[:5]}')
    print()
    print(f"{'set':<24}{'corr':>8}{'range':>8}{'sign%':>8}{'MAPE%':>8}{'n':>6}")
    out = {}
    for k, d in sets.items():
        s = score(d, src, common)
        out[k] = s
        print(f"{k:<24}{s['corr'].mean():8.3f}{s['range'].mean():8.3f}"
              f"{s['sign'].mean()*100:8.1f}{s['mape'].mean():8.1f}{len(s):6d}")
    # paired test against the first set listed
    base = list(sets)[0]
    print()
    for k in list(sets)[1:]:
        a = out[base].set_index('key')['corr']
        b = out[k].set_index('key')['corr']
        ix = a.index.intersection(b.index)
        if len(ix) < 5:
            continue
        from scipy import stats
        try:
            p = stats.wilcoxon(a.loc[ix], b.loc[ix]).pvalue
        except Exception:
            p = float('nan')
        d_ = b.loc[ix].mean() - a.loc[ix].mean()
        print(f'  {k} vs {base}: corr delta {d_:+.3f}  '
              f'wins {int((b.loc[ix] > a.loc[ix]).sum())}/{len(ix)}  p={p:.4f}')


if __name__ == '__main__':
    main()

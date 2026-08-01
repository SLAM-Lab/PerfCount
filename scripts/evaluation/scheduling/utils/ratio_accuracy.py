#!/usr/bin/env python3
"""Score a prediction directory in the DECISION-RELEVANT domain.

A cross-config translator predicts a speedup RATIO. The prediction files store a time,
`T_c = T_src / speedup_c`, where `T_src` is the true measured source-config time. `T_src` is
common to every configuration, so it cancels exactly in `argmin_c P_c T_c^(p+1)` -- it carries
most of the variance of the stored time but none of the decision. Reporting MAPE on the time
therefore credits a model for a quantity it never predicted and cannot act through.

Everything here is computed on the ratio itself:
  corr    Pearson correlation with the true ratio. This is the decision signal.
  range   std(predicted) / std(true). For an MSE-optimal predictor this should equal corr;
          materially above it means the model emits more variance than its skill justifies,
          i.e. it is injecting noise into the argmin.
  sign    accuracy of sign(ratio - 1), the crossover question, on samples where the true
          ratio is not within `--deadband` of 1 (near-ties are not decision-relevant).
  mape    on the ratio, for continuity with existing reporting.

Usage:
  ratio_accuracy.py <pred_dir> [<pred_dir> ...] [--src P_3.0GHz] [--limit 45]
"""
import argparse, os, sys, glob
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xproc_eval import GRAN, EXCLUDE, MIN_SAMPLES


def score_dir(pred_dir, src, limit=None, deadband=0.02):
    d = os.path.join(pred_dir, f'speedups_from_{src}')
    if not os.path.isdir(d):
        return None
    rows = []
    files = sorted(glob.glob(f'{d}/spec_*.csv'))
    files = [f for f in files if not os.path.basename(f)[:-4].startswith(EXCLUDE)][:limit]
    for f in files:
        wl = os.path.basename(f)[:-4]
        g = os.path.join(GRAN, f'speedups_{src}_{wl}.csv')
        if not os.path.exists(g):
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
            dec = np.abs(tv - 1.0) > deadband          # decision-relevant samples only
            sign = (np.sign(pv[dec] - 1.0) == np.sign(tv[dec] - 1.0)).mean() if dec.sum() else np.nan
            rows.append({'cfg': col.split('_vs_')[0].replace('Speedup_', ''),
                         'corr': np.corrcoef(pv, tv)[0, 1],
                         'range': pv.std() / max(tv.std(), 1e-12),
                         'sign': sign,
                         'mape': np.mean(np.abs(pv - tv) / tv) * 100,
                         'base_rate': (tv[dec] > 1.0).mean() if dec.sum() else np.nan})
    return pd.DataFrame(rows) if rows else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dirs', nargs='+')
    ap.add_argument('--src', default='P_3.0GHz')
    ap.add_argument('--limit', type=int, default=45)
    ap.add_argument('--deadband', type=float, default=0.02)
    ap.add_argument('--per_config', action='store_true')
    a = ap.parse_args()

    print(f'source {a.src}, ratio domain, deadband {a.deadband}\n')
    print(f"{'prediction set':<44}{'corr':>7}{'range':>7}{'sign%':>7}{'MAPE%':>7}{'n':>6}")
    for d in a.dirs:
        s = score_dir(d, a.src, a.limit, a.deadband)
        if s is None:
            print(f'{os.path.basename(d.rstrip("/")):<44}{"-- no data --":>34}')
            continue
        print(f'{os.path.basename(d.rstrip("/")):<44}{s["corr"].mean():7.3f}{s["range"].mean():7.3f}'
              f'{s["sign"].mean()*100:7.1f}{s["mape"].mean():7.1f}{len(s):6d}')
        if a.per_config:
            for c, g in s.groupby('cfg'):
                print(f'    {c:<40}{g["corr"].mean():7.3f}{g["range"].mean():7.3f}'
                      f'{g["sign"].mean()*100:7.1f}{g["mape"].mean():7.1f}{len(g):6d}')
    print('\nrange should equal corr for an MSE-optimal predictor; '
          'materially higher means the model is injecting noise into the argmin.')


if __name__ == '__main__':
    main()

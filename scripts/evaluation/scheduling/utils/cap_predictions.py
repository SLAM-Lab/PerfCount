#!/usr/bin/env python3
"""cap_predictions.py -- clamp predicted Speedups to within +/- CAP% of the TRUE
(oracle) speedup, generating one output dir per cap in a single pass.

Isolates model/forecast accuracy from the scheduling policy: CAP=0 -> oracle;
larger CAP lets the model's own error back in. Handles cross-freq
(speedups_<pfx>_<src>_<bench>_phase<p>.csv) and cross-proc (<bench>_phase<p>.csv).

Usage:
  cap_predictions.py --pred_dir D --granular G --out_base B --caps 5 10 20 --workers 16
  # writes B_cap5/, B_cap10/, B_cap20/
"""
import argparse, glob, os, re
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np, pandas as pd

SRCRE = re.compile(r'speedups_from_([PE])_([\d.]+GHz)$')
SPD = re.compile(r'Speedup_[PE]_[\d.]+GHz_vs_[PE]_[\d.]+GHz')


def process(f, pfx, src, granular, sub_name, out_bases, caps):
    fn = os.path.basename(f); stem = fn[:-4]
    mp = re.search(r'_phase(\d+)$', stem)
    if not mp: return 0
    phase = mp.group(1)
    pre = f'speedups_{pfx}_{src}_'
    bench = stem[len(pre):-len(f'_phase{phase}')] if stem.startswith(pre) else stem[:-len(f'_phase{phase}')]
    oracle = os.path.join(granular, f'speedups_{pfx}_{src}_{bench}_phase{phase}.csv')
    df = pd.read_csv(f)
    spd_cols = [c for c in df.columns if SPD.match(c)]
    ovals = {}
    if os.path.exists(oracle) and spd_cols:
        od = pd.read_csv(oracle, usecols=lambda c: c in spd_cols)
        for c in spd_cols:
            if c in od.columns:
                ovals[c] = od[c].values
    for cap, base in zip(caps, out_bases):
        p = cap / 100.0
        out = df.copy()
        for c, o in ovals.items():
            n = min(len(out), len(o))
            out.loc[:n-1, c] = np.clip(out[c].values[:n], o[:n]*(1-p), o[:n]*(1+p))
        d = os.path.join(base, sub_name); os.makedirs(d, exist_ok=True)
        out.to_csv(os.path.join(d, fn), index=False)
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred_dir', required=True)
    ap.add_argument('--granular', required=True)
    ap.add_argument('--out_base', required=True)
    ap.add_argument('--caps', type=float, nargs='+', required=True)
    ap.add_argument('--workers', type=int, default=16)
    a = ap.parse_args()
    out_bases = [f'{a.out_base}_cap{int(c)}' for c in a.caps]
    jobs = []
    for sub in sorted(glob.glob(os.path.join(a.pred_dir, 'speedups_from_*'))):
        m = SRCRE.search(os.path.basename(sub))
        if not m: continue
        for f in glob.glob(os.path.join(sub, '*.csv')):
            jobs.append((f, m.group(1), m.group(2), a.granular, os.path.basename(sub), out_bases, a.caps))
    n = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(process, *j) for j in jobs]
        for fut in as_completed(futs):
            n += fut.result()
    print(f"capped {n} pred files -> {', '.join(out_bases)}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Margin-gate a corrected prediction set against its baseline.

The hybrid classifier redirects samples toward the other core. Most of those redirects are
noise: at a 0.05 margin fewer than 2% of them survive, yet they carry nearly all of the
benefit, and the discarded ones are what produce the hybrid's losing record on homogeneous
workloads.

This writes a third prediction set that follows the corrected predictions only where the
correction claims an improvement above `--margin`, and the baseline everywhere else. The
claim is computed from the predictions and the characterised power table alone, so the rule
is runtime-observable. Truth is never read.

Because the claim is a metric-weighted cost ratio, the accepted sample set depends on the
metric (measured Jaccard between the EDP and ED2P sets is 26%). Generate one directory per
metric and read each metric's rows from its own simulator run.

Usage:
  gate_predictions.py --baseline <dir> --corrected <dir> --out <dir> --margin 0.05 --p_exp 1
"""
import argparse, os, sys, glob
import numpy as np, pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'utils'))
from xproc_eval import POW


def _prepare(base_csv, corr_csv, p_exp):
    """Shared decision arithmetic. Returns (baseline df, corrected df, n, flip, claim)."""
    b = pd.read_csv(base_csv)
    c = pd.read_csv(corr_csv)
    if list(b.columns) != list(c.columns):
        raise ValueError(f'column mismatch: {base_csv} vs {corr_csv}')
    n = min(len(b), len(c))
    b, c = b.iloc[:n].reset_index(drop=True), c.iloc[:n].reset_index(drop=True)

    tcol = [x for x in b.columns if x.startswith('Time_')]
    if len(tcol) != 1:
        raise ValueError(f'expected exactly one Time_ column in {base_csv}, got {tcol}')
    src = tcol[0][len('Time_'):]
    scols = [x for x in b.columns if x.startswith('Speedup_')]
    cfgs = [src] + [x[len('Speedup_'):].split('_vs_')[0] for x in scols]
    missing = [x for x in cfgs if x not in POW]
    if missing:
        raise KeyError(f'no characterised power for {missing} in {base_csv}')
    pv = np.array([POW[x] for x in cfgs])

    def times(df):
        t0 = df[tcol[0]].values.astype(float)
        out = [t0]
        for s in scols:
            v = df[s].values.astype(float)
            out.append(t0 / np.where(v > 1e-9, v, np.nan))
        return np.vstack(out)

    def dec(df):
        d = pv[:, None] * times(df) ** (p_exp + 1)
        return np.where(np.isfinite(d), d, np.inf)

    db, dc = dec(b), dec(c)
    idx = np.arange(n)
    cb, cc = np.argmin(db, axis=0), np.argmin(dc, axis=0)
    flip = cb != cc
    # Improvement the correction claims over the baseline's pick, in the correction's own
    # predicted units. Observable at runtime; never compared against truth.
    claim = (dc[cb, idx] - dc[cc, idx]) / np.maximum(dc[cb, idx], 1e-30)
    return b, c, n, flip, claim


def claims_only(base_csv, corr_csv, p_exp):
    """Claim values on flipped rows only, for calibrating a percentile threshold."""
    _, _, _, flip, claim = _prepare(base_csv, corr_csv, p_exp)
    return claim[flip]


def gate_file(base_csv, corr_csv, margin, p_exp):
    """Row-wise choice between two prediction frames. Returns (frame, n_flip, n_accept, n)."""
    b, c, n, flip, claim = _prepare(base_csv, corr_csv, p_exp)
    accept = flip & (claim > margin)

    out = b.copy()
    out.loc[accept, :] = c.loc[accept, :].values
    return out, int(flip.sum()), int(accept.sum()), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline', required=True)
    ap.add_argument('--corrected', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--margin', type=float, default=0.05)
    ap.add_argument('--accept_frac', type=float, default=None,
                    help='Calibrate the threshold so this fraction of ALL samples is accepted, '
                         'instead of using a fixed --margin. A fixed margin is not comparable '
                         'across metrics: ED2P weights cost by T^3, inflating relative gaps, so '
                         'the same margin accepts 82%% of flips under ED2P against 55%% under EDP. '
                         'Matching the accepted fraction is what makes the two arms test the same '
                         'hypothesis. The threshold is a single global constant calibrated offline, '
                         'so it stays deployable.')
    ap.add_argument('--p_exp', type=int, default=1, choices=[1, 2],
                    help='1 for EDP, 2 for ED2P. Determines which samples are accepted.')
    a = ap.parse_args()

    subs = sorted(os.path.basename(d) for d in glob.glob(f'{a.baseline}/speedups_from_*'))
    if not subs:
        sys.exit(f'no speedups_from_* under {a.baseline}')

    margin = a.margin
    if a.accept_frac is not None:
        # Pass 1: collect the claim distribution over flipped samples to set one global
        # threshold. Per-file thresholds would silently adapt to each phase, which is not
        # something a deployed scheduler could reproduce.
        cl, tot = [], 0
        for s in subs:
            for bf in sorted(glob.glob(f'{a.baseline}/{s}/*.csv')):
                cf = f'{a.corrected}/{s}/{os.path.basename(bf)}'
                if not os.path.exists(cf):
                    sys.exit(f'missing corrected file: {cf}')
                _, _, n, flip, claim = _prepare(bf, cf, a.p_exp)
                cl.append(claim[flip].astype(np.float32)); tot += n
        cl = np.concatenate(cl)
        keep = a.accept_frac * tot            # how many samples we are allowed to accept
        if keep >= len(cl):
            margin = -np.inf
        else:
            margin = float(np.quantile(cl, 1.0 - keep / len(cl)))
        print(f'calibrated margin={margin:.6f} for accept_frac={a.accept_frac} '
              f'(p_exp={a.p_exp}, {len(cl)} flips over {tot} samples)')
    tot = acc = flp = files = 0
    for s in subs:
        os.makedirs(f'{a.out}/{s}', exist_ok=True)
        for bf in sorted(glob.glob(f'{a.baseline}/{s}/*.csv')):
            cf = f'{a.corrected}/{s}/{os.path.basename(bf)}'
            if not os.path.exists(cf):
                sys.exit(f'missing corrected file: {cf}')
            df, nf, na, n = gate_file(bf, cf, margin, a.p_exp)
            df.to_csv(f'{a.out}/{s}/{os.path.basename(bf)}', index=False)
            tot += n; acc += na; flp += nf; files += 1
    print(f'{files} files  margin={margin:.6f}  p_exp={a.p_exp}\n'
          f'  samples {tot}\n'
          f'  correction flips argmin : {flp} ({flp/max(tot,1)*100:.3f}%)\n'
          f'  accepted after gate     : {acc} ({acc/max(tot,1)*100:.3f}%)  '
          f'= {acc/max(flp,1)*100:.1f}% of flips')


if __name__ == '__main__':
    main()

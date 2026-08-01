#!/usr/bin/env python3
"""How much does scoring "stay put" with ground truth bias a policy toward staying?

data_loader fills the diagonal of the prediction tensor (src_cfg == tgt_cfg) with oracle time.
A forecasting policy therefore evaluates the configuration it is already on using the TRUE cost
of the chunk it has not run yet, while evaluating every alternative with a forecast. Truth beats
a forecast on average, so the option scored with truth wins more often than it should. That is
an argmin bias toward staying, and the deployable policies do park.

This measures the size of it without changing anything. Two greedy runs over the same trace,
identical except for how the incumbent configuration is scored:

  A  stay = TRUE next-chunk cost      move = forecast     (what the simulator does today)
  B  stay = previous chunk's cost     move = forecast     (what a scheduler could actually know)

B is the honest counterpart: persistence is precisely the estimate the reactive policy uses, so
both options are then scored with a quantity available at decision time. Reported: how often the
choice differs, how often A stays where B moves, the residency gap, and the realized cost of
each under the true per-sample cost.

Myopic and transition-free, like the other offline probes here, so it sizes the bias rather than
predicting a simulator delta.

Usage: measure_diagonal_bias.py <forecast_dir> [--metric EDP] [--limit 40]
"""
import argparse, glob, os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xproc_eval import GRAN, POW, EXCLUDE, MIN_SAMPLES

FREQS = ['1.0', '2.0', '3.0', '4.0']
CONFIGS = [f'{p}_{f}GHz' for p in ('P', 'E') for f in FREQS]


def true_cost(wl, src, p_exp):
    """(cost[cfg, chunk], time[cfg, chunk]) from one granular trace. None if unusable."""
    g = os.path.join(GRAN, f'speedups_{src}_{wl}.csv')
    if not os.path.exists(g):
        return None
    t = pd.read_csv(g)
    if len(t) < MIN_SAMPLES:
        return None
    T0 = t[f'Time_{src}'].values.astype(float)
    T, P = [], []
    for c in CONFIGS:
        if c == src:
            T.append(T0)
        else:
            col = f'Speedup_{c}_vs_{src}'
            if col not in t.columns:
                return None
            s = t[col].values.astype(float)
            T.append(T0 / np.where(s > 1e-9, s, np.nan))
        pc = f'Power_{c}'
        if pc not in t.columns:
            return None
        P.append(t[pc].values.astype(float))
    T, P = np.vstack(T), np.vstack(P)
    C = P * T ** (p_exp + 1)
    return np.where(np.isfinite(C), C, np.inf), T


def forecast_times(fdir, wl, p_exp):
    """pred_time[src][cfg, chunk] from a forecast tensor directory, or None."""
    out = {}
    for src in CONFIGS:
        f = os.path.join(fdir, f'speedups_from_{src}', f'{wl}.csv')
        if not os.path.exists(f):
            f = os.path.join(fdir, f'speedups_from_{src}', f'speedups_{src}_{wl}.csv')
        if not os.path.exists(f):
            continue
        d = pd.read_csv(f)
        tcol = f'Time_{src}'
        if tcol not in d.columns:
            continue
        T0 = d[tcol].values.astype(float)
        m = {}
        for c in CONFIGS:
            if c == src:
                continue
            col = f'Speedup_{c}_vs_{src}'
            if col in d.columns:
                s = d[col].values.astype(float)
                m[c] = T0 / np.where(s > 1e-9, s, np.nan)
        if m:
            out[src] = m
    return out or None


def greedy(cost, pred, pv, n, stay_mode):
    """Greedy myopic run. stay_mode 'truth' scores the incumbent with the true next-chunk
    cost; 'persist' scores it with the previous chunk's realized cost."""
    idx = {c: i for i, c in enumerate(CONFIGS)}
    cur = 'P_3.0GHz'
    total, resid = 0.0, []
    for i in range(n):
        best, bestc = None, cur
        for c in CONFIGS:
            if c == cur:
                if stay_mode == 'truth':
                    v = cost[idx[c], i]
                else:
                    v = cost[idx[c], i - 1] if i > 0 else cost[idx[c], i]
            else:
                pm = pred.get(cur, {}).get(c)
                if pm is None or i >= len(pm) or not np.isfinite(pm[i]):
                    continue
                v = pv[idx[c]] * pm[i] ** 2      # decision power is the characterised table
            if np.isfinite(v) and (best is None or v < best):
                best, bestc = v, c
        total += cost[idx[bestc], i] if np.isfinite(cost[idx[bestc], i]) else 0.0
        resid.append(bestc)
        cur = bestc
    return total, resid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('forecast_dir')
    ap.add_argument('--metric', default='EDP', choices=['EDP', 'ED2P'])
    ap.add_argument('--limit', type=int, default=40)
    ap.add_argument('--src', default='P_3.0GHz')
    a = ap.parse_args()
    p_exp = 1 if a.metric == 'EDP' else 2
    pv = np.array([POW[c] for c in CONFIGS])

    files = sorted(os.path.basename(x)[:-4]
                   for x in glob.glob(f'{a.forecast_dir}/speedups_from_{a.src}/spec_*.csv'))
    files = [w for w in files if not w.startswith(EXCLUDE)][:a.limit]
    rows = []
    for wl in files:
        tc = true_cost(wl, a.src, p_exp)
        if tc is None:
            continue
        cost, _ = tc
        pred = forecast_times(a.forecast_dir, wl, p_exp)
        if pred is None:
            continue
        n = min(cost.shape[1], min(len(v) for m in pred.values() for v in m.values()))
        if n < MIN_SAMPLES:
            continue
        ta, ra = greedy(cost, pred, pv, n, 'truth')
        tb, rb = greedy(cost, pred, pv, n, 'persist')
        opt = np.nanargmin(np.where(np.isfinite(cost[:, :n]), cost[:, :n], np.inf), axis=0)
        stay_a = np.mean([ra[i] == ra[i - 1] for i in range(1, n)])
        stay_b = np.mean([rb[i] == rb[i - 1] for i in range(1, n)])
        rows.append({'wl': wl, 'n': n,
                     'differ': np.mean([ra[i] != rb[i] for i in range(n)]),
                     'A_stayed_B_moved': np.mean([ra[i] == ra[i - 1] and rb[i] != rb[i - 1]
                                                  for i in range(1, n)]),
                     'A_hold_rate': stay_a, 'B_hold_rate': stay_b,
                     'oracle_hold_rate': np.mean(opt[1:] == opt[:-1]),
                     'A_cost': ta, 'B_cost': tb})
    if not rows:
        sys.exit('no usable phases')
    d = pd.DataFrame(rows)
    print(f'{a.metric}, source {a.src}, {len(d)} phases. '
          f'A = stay scored with truth (current sim); B = stay scored with persistence.\n')
    print(f"  chunks where the two choose differently : {d.differ.mean()*100:6.2f}%")
    print(f"  A held while B moved                    : {d.A_stayed_B_moved.mean()*100:6.2f}%")
    print(f"  hold rate  A {d.A_hold_rate.mean()*100:6.2f}%   "
          f"B {d.B_hold_rate.mean()*100:6.2f}%   oracle {d.oracle_hold_rate.mean()*100:6.2f}%")
    rel = (d.B_cost - d.A_cost) / d.A_cost * 100
    print(f"  realized cost B vs A                    : {rel.mean():+6.2f}% "
          f"(negative = the honest scoring is cheaper)")
    gap = (d.A_hold_rate.mean() - d.B_hold_rate.mean()) * 100
    verb = 'raises' if gap > 0 else 'LOWERS'
    print(f"\n  A's truth-scored diagonal {verb} the hold rate by {abs(gap):.2f} pp "
          f"against an oracle hold rate of {d.oracle_hold_rate.mean()*100:.2f}%.")
    if gap <= 0:
        print("  So the truth diagonal is NOT the source of the parking: scoring the incumbent\n"
              "  with an estimate instead makes the policy hold MORE, not less. Truth on the\n"
              "  incumbent fluctuates with the real signal, so it sometimes looks bad enough to\n"
              "  leave; a lagged estimate rarely does. Persistence is not a self-forecast, so\n"
              "  this rules out one alternative rather than the symmetric ideal.")


if __name__ == '__main__':
    main()

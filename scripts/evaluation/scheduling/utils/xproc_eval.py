#!/usr/bin/env python3
"""Shared offline evaluation harness for cross-processor prediction sets.

Scores a prediction directory by the myopic per-sample argmin on the crossover slice
(the source config plus the four configurations of the other core), against the true
per-sample optimum from the granular traces. No transition costs and no policy dynamics,
so numbers here are an upper bound on what a scheduler can realise -- the point is to
compare prediction sets against each other under one fixed, cheap scorer, not to predict
the simulator.

Every consumer of this module shares one loader and one scorer so that the pieces built on
top (gating, deconfounding, frequency discrimination) cannot drift apart in their
definitions. `selftest` regression-checks the scorer against known values.
"""
import os, glob
import numpy as np, pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
HP = os.path.join(REPO, 'results/scheduling/Hetero_precompute')
GRAN = os.path.join(HP, 'speedup_full_v2_repaired/granular_phase_traces')

# Characterised per-configuration power (data_loader.POWER_W). A policy may use only this,
# never the per-sample truth, to choose. Scoring always uses the per-sample truth.
POW = {'P_1.0GHz': 1.83, 'P_2.0GHz': 3.75, 'P_3.0GHz': 6.86, 'P_4.0GHz': 14.05,
       'E_1.0GHz': 1.29, 'E_2.0GHz': 2.47, 'E_3.0GHz': 5.40, 'E_4.0GHz': 17.64}

# Two workloads ~900x shorter than every other SPEC2026 benchmark; excluded from every
# per-workload breakdown in the chapter, so excluded here too.
EXCLUDE = ('spec_772.marian_r', 'spec_706.stockfish_r')
MIN_SAMPLES = 200


def slice_configs(src):
    """The crossover action space reachable from source config `src`."""
    other = 'E' if src.startswith('P') else 'P'
    return [src] + [f'{other}_{f}GHz' for f in ('1.0', '2.0', '3.0', '4.0')]


def load_truth(wl, src):
    """True per-sample time and power for every config in the slice. None if unusable."""
    g = os.path.join(GRAN, f'speedups_{src}_{wl}.csv')
    if not os.path.exists(g):
        return None
    t = pd.read_csv(g)
    if len(t) < MIN_SAMPLES:
        return None
    cfgs = slice_configs(src)
    T0 = t[f'Time_{src}'].values
    T, P = [T0], [t[f'Power_{src}'].values]
    for c in cfgs[1:]:
        sp, pw = t.get(f'Speedup_{c}_vs_{src}'), t.get(f'Power_{c}')
        if sp is None or pw is None:
            return None
        T.append(T0 / np.where(sp.values > 1e-9, sp.values, np.nan))
        P.append(pw.values)
    return np.vstack(T), np.vstack(P)


def load_pred(pred_dir, wl, src):
    """Predicted per-sample time for every config in the slice. None if absent."""
    f = os.path.join(pred_dir, f'speedups_from_{src}', f'{wl}.csv')
    if not os.path.exists(f):
        return None
    p = pd.read_csv(f)
    T0 = p[f'Time_{src}'].values
    out = [T0]
    for c in slice_configs(src)[1:]:
        col = f'Speedup_{c}_vs_{src}'
        if col not in p.columns:
            return None
        s = p[col].values
        out.append(T0 / np.where(s > 1e-9, s, np.nan))
    return np.vstack(out)


def cost(power, times, p_exp):
    """Metric cost per config per sample: E*T^p = (P*T)*T^p.

    `power` is either a per-config vector (the characterised table, what a decision may use)
    or a per-config-per-sample matrix (the truth, what scoring uses).
    """
    p = np.asarray(power)
    if p.ndim == 1:
        p = p[:, None]
    return p * times ** (p_exp + 1)


def phases(pred_dir, src, suite_prefix='spec_'):
    """Workload-phase names present in a prediction directory, excluding the short pair."""
    d = os.path.join(pred_dir, f'speedups_from_{src}')
    names = sorted(os.path.basename(x)[:-4] for x in glob.glob(f'{d}/{suite_prefix}*.csv'))
    return [w for w in names if not w.startswith(EXCLUDE)]


def score_phase(truth, preds, src, p_exp, choose=None):
    """Realised/oracle for each named prediction set on one phase.

    `preds` maps name -> predicted time matrix. `choose` optionally overrides the decision
    rule; it receives (name, decision_cost_matrix, baseline_choice) and returns an index
    array. Returns (dict of ratios, dict of diagnostics, true optimal choice array).
    """
    Tt, Pt = truth
    n = min([Tt.shape[1]] + [v.shape[1] for v in preds.values()])
    Tt, Pt = Tt[:, :n], Pt[:, :n]
    cfgs = slice_configs(src)
    pv = np.array([POW[c] for c in cfgs])
    idx = np.arange(n)

    true = cost(Pt, Tt, p_exp)          # scoring uses per-sample truth
    true = np.where(np.isfinite(true), true, np.inf)
    opt = np.nanargmin(true, axis=0)
    den = true[opt, idx].sum()

    ratios, diag, base_choice = {}, {}, None
    for name, pm in preds.items():
        dec = cost(pv, pm[:, :n], p_exp)   # decisions use characterised power only
        dec = np.where(np.isfinite(dec), dec, np.inf)
        ch = np.nanargmin(dec, axis=0)
        if base_choice is None:
            base_choice = ch
        if choose is not None:
            ch = choose(name, dec, base_choice, ch)
        ratios[name] = true[ch, idx].sum() / den
        diag[name] = {'other_core_frac': float(np.mean(ch > 0)),
                      'switch_rate': float(np.mean(ch[1:] != ch[:-1]))}
    diag['_truth'] = {'other_core_frac': float(np.mean(opt > 0)),
                      'switch_rate': float(np.mean(opt[1:] != opt[:-1])),
                      'offmodal': float(1 - np.bincount(opt, minlength=len(cfgs)).max() / n),
                      'n': int(n)}
    return ratios, diag, opt


def evaluate(pred_dirs, src='P_3.0GHz', p_exp=1, choose=None, limit=None):
    """Score several prediction directories over every shared phase. Returns a DataFrame."""
    names = list(pred_dirs)
    common = set(phases(pred_dirs[names[0]], src))
    for k in names[1:]:
        common &= set(phases(pred_dirs[k], src))
    common = sorted(common)[:limit]
    rows = []
    for wl in common:
        truth = load_truth(wl, src)
        if truth is None:
            continue
        preds, ok = {}, True
        for k in names:
            pm = load_pred(pred_dirs[k], wl, src)
            if pm is None:
                ok = False
                break
            preds[k] = pm
        if not ok:
            continue
        ratios, diag, _ = score_phase(truth, preds, src, p_exp, choose)
        r = {'wl': wl, 'n': diag['_truth']['n'], 'offmodal': diag['_truth']['offmodal'],
             'true_other': diag['_truth']['other_core_frac']}
        for k in names:
            r[k] = ratios[k]
            r[f'{k}_other'] = diag[k]['other_core_frac']
            r[f'{k}_switch'] = diag[k]['switch_rate']
        rows.append(r)
    return pd.DataFrame(rows)


def split(df, key='offmodal'):
    """Median split on decision-relevant heterogeneity, measured from the oracle's choices."""
    thr = df[key].median()
    return df[df[key] > thr], df[df[key] <= thr], thr


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'selftest':
        B = os.path.join(HP, 'cross_proc_translate_10M')
        H = os.path.join(HP, 'cross_proc_translate_hybrid_10M')
        ok = True
        for p_exp, met, exp_b, exp_h, exp_n in ((1, 'EDP', 1.0840, 1.0788, 120),
                                                (2, 'ED2P', 1.1156, 1.1072, 120)):
            d = evaluate({'base': B, 'hyb': H}, p_exp=p_exp)
            b, h = d['base'].mean(), d['hyb'].mean()
            good = (abs(b - exp_b) < 5e-4 and abs(h - exp_h) < 5e-4 and len(d) == exp_n)
            ok &= good
            print(f"{met:5s} n={len(d):3d} (exp {exp_n})  base={b:.4f} (exp {exp_b})  "
                  f"hyb={h:.4f} (exp {exp_h})  {'PASS' if good else 'FAIL'}")
        het, hom, thr = split(evaluate({'base': B, 'hyb': H}, p_exp=1))
        print(f"split: het n={len(het)} base={het['base'].mean():.4f} (exp 1.1577) | "
              f"hom n={len(hom)} base={hom['base'].mean():.4f} (exp 1.0103)")
        sys.exit(0 if ok else 1)
    print(__doc__)

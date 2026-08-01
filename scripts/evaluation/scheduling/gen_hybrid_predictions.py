#!/usr/bin/env python3
"""Crossover-corrected cross-processor predictions.

The translation regressor answers the per-chunk question "which config wins" with the
workload average: it detects ~8% of genuine E-core wins and in a third of cases its
predictions never cross the tie-point at all. That is a property of the squared-error
objective, which is minimised by the conditional mean. A classifier trained on the same
counters recovers the signal (cross-workload AUC 0.73 with data-driven counters).

This emits corrected prediction directories that the simulator consumes unchanged, so every
policy (reactive, gated forecast, perfect future) picks the correction up.

Gating. A headroom-blind override is worthless in aggregate (+0.11% over 90 phases) because
the classifier is only reliable where the decision is far from right (precision 88% on
high-headroom phases, 44% on near-optimal ones). Headroom is not observable at runtime, so
we gate on the deployable surrogate: the model's own predicted benefit margin.

Usage: gen_hybrid_predictions.py --in_dir D --out_dir D2 [--p_thresh 0.5] [--margin 0.05]
"""
import argparse, glob, os, re, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingClassifier as HGB

# Anchored to the repo root, not the CWD. These were relative paths, which silently resolved
# to nothing when the script was invoked from its own directory: every lookup missed, the
# correction was skipped for every file, and the run still exited 0 having written an exact
# copy of its input. See the missing-input accounting in main().
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     '..', '..', '..'))
TR = os.path.join(_ROOT, 'results/scheduling/Hetero_precompute/'
                         'speedup_full_v2_repaired/granular_phase_traces')
PMU = os.path.join(_ROOT, 'processed_data_10M_power/x86_desktop_heterogeneous')
POW = {'P_1.0GHz':1.83,'P_2.0GHz':3.75,'P_3.0GHz':6.86,'P_4.0GHz':14.05,
       'E_1.0GHz':1.29,'E_2.0GHz':2.47,'E_3.0GHz':5.40,'E_4.0GHz':17.64}
# Data-driven selection gave 8 counters, but l1_dcache_load_misses, mem_loads_aux and
# uops_executed_thread are exposed only by the P-core (Golden Cove) and not by the E-core
# (Gracemont). A model that consumes them cannot run when the workload sits on an E-core, and
# zero-filling them there would feed the model fabricated inputs. We therefore use the subset
# available on BOTH core types, so one model works from either source.
# Sentinel: inputs are present but the trace is too short to fit on. Distinct from None
# (inputs genuinely missing), which is the condition worth aborting over.
TOO_SHORT = 'too_short'

TOP8 = ['ref_cycles', 'llc_misses', 'branch_load_misses', 'dtlb_store_misses', 'cache_references']


def feats(X, n):
    F = X.iloc[:n].select_dtypes(include=[np.number]).drop(columns=['sample_index'], errors='ignore')
    F = F.drop(columns=[c for c in F.columns if c.startswith('power_watts')], errors='ignore')
    inst = F['instructions'].replace(0, np.nan).values
    R = F.div(inst, axis=0).replace([np.inf, -np.inf], np.nan).fillna(0)
    R['ref_cycles'] = np.nan_to_num(F['ref_cycles'].values / inst)
    # Pin the column set: some workloads' counter files lack a few of these, and an
    # inconsistent feature frame between fit and predict is a silent-mismatch failure.
    missing = [c for c in TOP8 if c not in R.columns]
    if missing:
        raise KeyError(f'counter file missing required columns: {missing}')
    return R[TOP8].fillna(0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in_dir', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--p_thresh', type=float, default=0.5)
    ap.add_argument('--margin', type=float, default=0.05)
    ap.add_argument('--folds', type=int, default=4)
    a = ap.parse_args()

    srcs = sorted(glob.glob(f'{a.in_dir}/speedups_from_*'))
    items = []
    for sd in srcs:
        src = os.path.basename(sd).replace('speedups_from_', '')
        sf, sc = src.split('_')[1], ('0' if src[0] == 'P' else '16')
        for f in sorted(glob.glob(f'{sd}/*.csv')):
            m = re.match(r'(.+)_phase(\d+)$', os.path.basename(f)[:-4])
            if not m:
                continue
            wl, ph = m.group(1), m.group(2)
            items.append((src, sf, sc, wl, ph, f, sd))
    wls = sorted({i[3] for i in items})
    fold = {w: k % a.folds for k, w in enumerate(wls)}
    print(f'{len(items)} files, {len(wls)} workloads, {a.folds} folds', flush=True)
    for p, what in ((TR, 'granular traces'), (PMU, 'counter files')):
        if not os.path.isdir(p):
            raise SystemExit(f'{what} directory does not exist: {p}')

    # build training data per fold: label = does ANY E config beat the source this chunk
    cache = {}
    def get(src, sf, sc, wl, ph, f):
        key = (src, wl, ph)
        if key in cache:
            return cache[key]
        t = f'{TR}/speedups_{src}_{wl}_phase{ph}.csv'
        p = f'{PMU}/aligned_{wl}_{sf}_cpu{sc}_phase{ph}.csv'
        if not (os.path.exists(t) and os.path.exists(p)):
            cache[key] = None; return None
        try:
            T = pd.read_csv(t); X = pd.read_csv(p); P = pd.read_csv(f)
        except Exception:
            cache[key] = None; return None
        cols = [c for c in P.columns if c.startswith('Speedup_') and c in T.columns]
        n = min(len(T), len(X), len(P))
        if n < 500 or not cols:
            # Distinct from a missing input: the files are present, the trace is just too
            # short to fit on (spec_706.stockfish_r and spec_772.marian_r are ~900x shorter
            # than every other workload and are excluded from the chapter's breakdowns).
            # The caller must not treat this as an anomaly.
            cache[key] = TOO_SHORT; return TOO_SHORT
        tgts = [re.match(r'Speedup_(.+)_vs_', c).group(1) for c in cols]
        base = T[f'Time_{src}'].values[:n]
        Tt = np.stack([base] + [base / np.maximum(T[c].values[:n], 1e-9) for c in cols], axis=1)
        cfg = [src] + tgts
        pv = np.array([POW.get(c, 5.0) for c in cfg])
        y = (pv * Tt ** 2).argmin(axis=1) != 0          # a non-source config wins this chunk
        cache[key] = (feats(X, n), y.astype(int), cols, tgts, n)
        return cache[key]

    os.makedirs(a.out_dir, exist_ok=True)
    for k in range(a.folds):
        Xs, ys = [], []
        for (src, sf, sc, wl, ph, f, sd) in items:
            if fold[wl] == k:
                continue
            g = get(src, sf, sc, wl, ph, f)
            if g and g is not TOO_SHORT:
                Xs.append(g[0].iloc[::6]); ys.append(g[1][::6])
        if not Xs:
            continue
        clf = HGB(max_iter=100, max_depth=5, random_state=0,
                  early_stopping=False).fit(pd.concat(Xs), np.concatenate(ys))
        wrote = skipped = corrected = 0
        unexpected = []
        for (src, sf, sc, wl, ph, f, sd) in items:
            if fold[wl] != k:
                continue
            g = get(src, sf, sc, wl, ph, f)
            od = f"{a.out_dir}/{os.path.basename(sd)}"
            os.makedirs(od, exist_ok=True)
            P = pd.read_csv(f)
            if g is None or g is TOO_SHORT:
                skipped += 1
                # DaCapo has no aligned power-counter files at all, so it cannot be
                # corrected and is expected here. A SPEC file arriving in this branch means
                # something is actually wrong with the inputs.
                if g is None and not wl.startswith('dacapo'):
                    unexpected.append(f'{src}/{wl}_phase{ph}')
            if g and g is not TOO_SHORT:
                corrected += 1
                Xf, _, cols, tgts, n = g
                pE = clf.predict_proba(Xf)[:, 1]
                sp = P[cols].values[:n]                       # speedup target-vs-source, >1 = faster
                pv_s = POW.get(src, 5.0)
                pv_t = np.array([POW.get(t_, 5.0) for t_ in tgts])
                cost_s = pv_s
                cost_t = pv_t[None, :] / np.maximum(sp, 1e-9) ** 2
                best = cost_t.argmin(axis=1)
                # Speedup the best target would need in order to beat the source.
                need = np.sqrt(pv_t[best] / cost_s) * 1.001
                cur = sp[np.arange(n), best]
                # The correction applies exactly where the regressor puts the target BEHIND
                # the source but the classifier says it wins.
                #
                # The gate bounds how far the prediction is moved, not the benefit the
                # regressor already sees. Gating on the regressor's own predicted benefit
                # (the previous rule) was self-defeating: it required the target to already
                # be winning, which is the complement of the set the edit applies to, so the
                # correction fired on exactly 0 of 547k samples. Bounding the nudge instead
                # keeps every edit near the tie-point, where flipping the decision is cheap
                # if the classifier is wrong.
                nudge = need / np.maximum(cur, 1e-9) - 1.0
                fire = (pE > a.p_thresh) & (cur < need) & (nudge <= a.margin)
                newsp = sp.copy()
                idx = np.where(fire)[0]
                newsp[idx, best[idx]] = need[idx]
                for j, c in enumerate(cols):
                    P.loc[:n - 1, c] = newsp[:, j]
            P.to_csv(f'{od}/{wl}_phase{ph}.csv', index=False)
            wrote += 1
        print(f'  fold {k}: wrote {wrote}  corrected {corrected}  '
              f'skipped-for-missing-inputs {skipped}', flush=True)
        # A file whose inputs are missing is written through unchanged. That is only
        # acceptable as an exception. If it is the rule, the output is a copy of the input
        # wearing the name of a corrected set, which is indistinguishable downstream from a
        # real result -- exactly the failure that produced a fake NEED-6 answer.
        if corrected == 0:
            raise SystemExit(
                f'fold {k}: every one of {wrote} files was written through UNCORRECTED '
                f'(missing traces or counter files). Refusing to emit a copy of --in_dir.')
        if unexpected:
            raise SystemExit(
                f'fold {k}: {len(unexpected)} non-DaCapo files lacked inputs and were '
                f'written through uncorrected, e.g. {unexpected[:3]}. DaCapo has no power '
                f'counters and is expected to skip; SPEC is not.')
    print('DONE', flush=True)


if __name__ == '__main__':
    main()

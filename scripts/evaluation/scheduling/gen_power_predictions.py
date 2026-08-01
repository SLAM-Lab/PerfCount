#!/usr/bin/env python3
"""Per-chunk predicted power for every configuration, for use as a deployable
decision-power estimate.

Absolute per-config power is not predictable from counters (R2 ~ 0 or negative). The
ratio of a target config's power to the *measured* power of the config currently running
is (R2 ~ 0.6 cross-workload), because the ratio is governed by the two operating points'
fixed voltage/frequency relationship rather than by workload activity. So we predict the
ratio and anchor it on a measured quantity:

    PredPower_c(i) = MeasuredPower_ref(i) * predicted_ratio_c(i)

A per-chunk ratio that VARIES across configs is the point. A constant ratio (which is what
the static POWER_W table encodes) cannot change any decision, since a per-chunk scalar
common to all configs factors out of argmin_c P_c * T_c^p.

Workloads are split into folds and each workload's predictions come from a model that never
saw it, so the output is usable for evaluation without leakage.

Usage: gen_power_predictions.py --out_dir DIR [--ref P_3.0GHz] [--folds 4]
"""
import argparse, glob, os, re, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingRegressor as HGR

TR = 'results/scheduling/Hetero_precompute/speedup_full_v2_repaired/granular_phase_traces'
PMU = 'processed_data_10M_power/x86_desktop_heterogeneous'
CONFIGS = ['P_1.0GHz','P_2.0GHz','P_3.0GHz','P_4.0GHz','E_1.0GHz','E_2.0GHz','E_3.0GHz','E_4.0GHz']


# Counter availability differs both between core types (Gracemont exposes fewer events than
# Golden Cove) and between workloads. Pin the feature set to counters present everywhere and
# fail loudly on a miss, rather than let an inconsistent frame reach the model.
# mem_loads is present for SPEC but absent from the DaCapo aligned counter files, so it is
# dropped to keep one consistent feature set across every suite the model trains on.
FEATS = ['branch_load_misses', 'branch_loads', 'branch_misses', 'branches', 'bus_cycles', 'cache_misses', 'cache_references', 'cpu_cycles', 'dtlb_load_misses', 'dtlb_loads', 'dtlb_store_misses', 'dtlb_stores', 'instructions', 'itlb_load_misses', 'l1_icache_load_misses', 'llc_loads', 'llc_misses', 'mem_stores', 'ref_cycles']


def feats(X, n):
    F = X.iloc[:n].select_dtypes(include=[np.number]).drop(columns=['sample_index'], errors='ignore')
    F = F.drop(columns=[c for c in F.columns if c.startswith('power_watts')], errors='ignore')
    inst = F['instructions'].replace(0, np.nan).values
    R = F.div(inst, axis=0).replace([np.inf, -np.inf], np.nan).fillna(0)
    R['ref_cycles'] = np.nan_to_num(F['ref_cycles'].values / inst)
    missing = [c for c in FEATS if c not in R.columns]
    if missing:
        raise KeyError(f'counter file missing required columns: {missing}')
    return R[FEATS].fillna(0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--ref', default='P_3.0GHz')
    ap.add_argument('--folds', type=int, default=4)
    # Anchor the level on the ref config's power LAG samples earlier, not the current chunk.
    # lag=0 reproduces the original (uses chunk i's ref power, not causal); lag=1 is the
    # deployable "last sample" persistence anchor. The first `lag` chunks reuse the earliest
    # available reading.
    ap.add_argument('--lag', type=int, default=0)
    # How the cross-config ratio is formed: 'model' = per-config HGR on counters (R2~0.6);
    # 'static' = a single scalar mean ratio per config from the training folds (the level is
    # still per-chunk from the anchor, so unlike the constant POWER_W table this DOES track
    # the workload's current power, but keeps a fixed cross-config shape).
    ap.add_argument('--ratio_mode', choices=['model', 'static'], default='model')
    # Process one test fold only (train on the rest). Folds partition workloads, so their output
    # phase files are disjoint and many --fold processes can write the same --out_dir in parallel.
    ap.add_argument('--fold', type=int, default=None)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    ref = a.ref
    rf, rc = ref.split('_')[1], ('0' if ref[0] == 'P' else '16')

    # Counter-feature files live in the SPEC power dir for SPEC and in a dacapo_c1 subdir for
    # DaCapo. Only ratio_mode=model needs them; the power targets (pw) always come from the
    # speedups trace, which carries Power_<config> for every suite including DaCapo.
    DACO = 'processed_data_10M/x86_desktop_heterogeneous/dacapo_c1/dacapo_c1_3ghz'
    def find_feat(wl, ph):
        for base in (PMU, DACO):
            q = f'{base}/aligned_{wl}_{rf}_cpu{rc}_phase{ph}.csv'
            if os.path.exists(q):
                return q
        return None

    items = []
    dropped_model = 0
    for t in sorted(glob.glob(f'{TR}/speedups_{ref}_*.csv')):
        m = re.match(rf'.*speedups_{ref}_(.+)_phase(\d+)\.csv', t)
        if not m:
            continue
        wl, ph = m.group(1), m.group(2)
        try:
            T = pd.read_csv(t)
        except Exception:
            continue
        cf = [c for c in CONFIGS if f'Power_{c}' in T.columns]
        if ref not in cf or len(cf) < 4:
            continue
        R = None
        if a.ratio_mode == 'model':
            p = find_feat(wl, ph)
            if p is None:
                dropped_model += 1; continue
            try:
                X = pd.read_csv(p); n = min(len(T), len(X))
                if n < 500:
                    continue
                R = feats(X, n)
            except (KeyError, Exception):
                dropped_model += 1; continue
        else:  # static ratios: no counter features required, so every suite is covered
            n = len(T)
            if n < 500:
                continue
        pw = {c: T[f'Power_{c}'].values[:n] for c in cf}
        items.append(dict(wl=wl, ph=ph, n=n, cf=cf, pw=pw, R=R))
    if dropped_model:
        print(f'  [note] dropped {dropped_model} phases lacking counter features (model mode)', flush=True)
    wls = sorted({i['wl'] for i in items})
    fold = {w: k % a.folds for k, w in enumerate(wls)}
    print(f'{len(items)} workload-phases, {len(wls)} workloads, {a.folds} folds, ref={ref}', flush=True)

    # assemble training rows per target config
    folds_to_run = [a.fold] if a.fold is not None else range(a.folds)
    for k in folds_to_run:
        tr_items = [i for i in items if fold[i['wl']] != k]
        te_items = [i for i in items if fold[i['wl']] == k]
        models = {}          # config -> HGR (ratio_mode=model)
        static_ratio = {}    # config -> scalar mean ratio (ratio_mode=static)
        for c in CONFIGS:
            Xs, ys = [], []
            for i in tr_items:
                if c not in i['cf']:
                    continue
                ok = (i['pw'][ref] > 0.5) & (i['pw'][c] > 0.5)
                if ok.sum() < 200:
                    continue
                if a.ratio_mode == 'model':          # features only needed to fit the ratio model
                    Xs.append(i['R'][ok].iloc[::4])
                ys.append((i['pw'][c][ok] / i['pw'][ref][ok])[::4])
            if not ys:
                continue
            y_all = np.concatenate(ys)
            static_ratio[c] = float(np.mean(y_all))
            if a.ratio_mode == 'model':
                models[c] = HGR(max_iter=100, max_depth=5, random_state=0).fit(pd.concat(Xs), y_all)

        def anchor(base):
            # ref power lagged by --lag samples (persistence); first `lag` reuse earliest reading.
            if a.lag <= 0:
                return base
            return np.concatenate([np.full(a.lag, base[0]), base[:-a.lag]])

        ratio_keys = models if a.ratio_mode == 'model' else static_ratio
        for i in te_items:
            out = {}
            base = anchor(i['pw'][ref])
            for c in CONFIGS:
                if c == ref:
                    out[f'PredPower_{c}'] = base
                elif c in ratio_keys:
                    r = (np.clip(models[c].predict(i['R']), 0.02, 20.0)
                         if a.ratio_mode == 'model' else static_ratio[c])
                    out[f'PredPower_{c}'] = r * base
            pd.DataFrame(out).to_csv(f"{a.out_dir}/{i['wl']}_phase{i['ph']}.csv", index=False)
        print(f'  fold {k}: wrote {len(te_items)} phases '
              f'({len(ratio_keys)} config {a.ratio_mode}s, lag={a.lag})', flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Decision-level diagnostics for the cross-platform / forecast prediction tensors.

The suite-level EDP numbers say a policy is bad. They do not say why. This tool
reconstructs exactly what a greedy policy sees at each chunk and reports the four
quantities that distinguish the candidate explanations:

  1. PREDICTION BIAS   per (source, target) config: median signed relative error of
                       predicted time. Answers "is the model systematically optimistic
                       about some core?"
  2. DECISION CONFUSION oracle's argmin vs model's argmin, as a matrix. Answers "when
                       the truth says P_3.0, what does the model pick instead?"
  3. REGRET DECOMPOSITION  per chunk, the EDP cost of the model's choice over the
                       oracle's, attributed to the config it wrongly chose. Answers
                       "which specific mistake costs the energy?"
  4. ABSORBING BEHAVIOUR  transition matrix over chosen configs. Answers "does the
                       policy get stuck somewhere it cannot leave?"

Usage:
  diagnose_predictions.py --traces <granular_phase_traces> \
      --cross_freq_p <dir> --cross_freq_e <dir> --cross_proc <dir> \
      [--forecast_p <dir> --forecast_e <dir> --forecast_cp <dir>] \
      [--suite spec26] [--bench spec_710.omnetpp_r] [--metric EDP] [--limit 40]

Compares the forecast tensor against the reactive tensor when both are supplied, which
is the setup for asking why forecasting fails to beat reactive on the heterogeneous
problem.
"""
import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
import data_loader  # noqa: E402
from data_loader import (ALL_MODEL_CONFIGS, load_phase_data,  # noqa: E402
                         _load_model_time_mat, _load_e_model_time_mat,
                         _load_cross_proc_time_mat, _load_full_model_time_mat)

CONFIGS = ['E_1.0GHz', 'E_2.0GHz', 'E_3.0GHz', 'E_4.0GHz',
           'P_1.0GHz', 'P_2.0GHz', 'P_3.0GHz', 'P_4.0GHz']


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


def discover(traces):
    """Return [(workload, phase)] present in the trace dir."""
    out = set()
    for f in Path(traces).glob('speedups_P_4.0GHz_*_phase*.csv'):
        stem = f.name.replace('speedups_P_4.0GHz_', '').replace('.csv', '')
        wl, ph = stem.rsplit('_phase', 1)
        out.add((wl, int(ph)))
    return sorted(out)


def edp(t, e, p):
    return e * t if p == 1 else e * t * t


def greedy_path(tensor_t, oracle_t, oracle_e, power, metric_p, start):
    """Replay a greedy policy over a source-keyed tensor. Returns chosen indices.

    Mirrors the simulator: at chunk i the policy reads tensor_t[src_of(prev), i, :],
    derives energy as predicted_time * true_power, and takes the argmin of the metric.
    """
    n = oracle_t.shape[0]
    chosen = np.empty(n, dtype=int)
    prev = start
    for i in range(n):
        src = prev
        row_t = tensor_t[src, i, :]
        row_e = row_t * power[i, :]
        score = edp(row_t, row_e, metric_p)
        score = np.where(np.isfinite(score), score, np.inf)
        a = int(np.argmin(score))
        chosen[i] = a
        prev = a
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--traces', required=True)
    ap.add_argument('--cross_freq_p', required=True)
    ap.add_argument('--cross_freq_e', required=True)
    ap.add_argument('--cross_proc', required=True)
    ap.add_argument('--forecast_p'); ap.add_argument('--forecast_e')
    ap.add_argument('--forecast_cp')
    ap.add_argument('--suite', default=None, help='spec17|spec26|dacapo')
    ap.add_argument('--bench', default=None)
    ap.add_argument('--metric', default='EDP', choices=['EDP', 'ED2P'])
    ap.add_argument('--limit', type=int, default=40, help='max phases')
    ap.add_argument('--max_chunks', type=int, default=20000)
    a = ap.parse_args()
    mp = 1 if a.metric == 'EDP' else 2
    traces = Path(a.traces)

    phases = discover(traces)
    if a.bench:
        phases = [p for p in phases if p[0] == a.bench]
    if a.suite:
        phases = [p for p in phases if suite_of(p[0]) == a.suite]
    phases = phases[:a.limit]
    if not phases:
        sys.exit('no phases matched')
    print(f'analyzing {len(phases)} phases, metric={a.metric}\n')

    bias = defaultdict(list)          # (src,tgt) -> [median signed rel err]
    confusion = Counter()             # (oracle_pick, model_pick) -> chunks
    regret = defaultdict(float)       # model_pick -> summed excess EDP
    picked = Counter()
    trans = Counter()                 # (from,to) -> count, model policy
    fc_confusion = Counter()
    fc_picked = Counter()
    n_chunks_tot = 0

    for wl, ph in phases:
        d = load_phase_data(wl, ph, traces, CONFIGS, power_mode='per_sample',
                            model_pred_dir=Path(a.cross_freq_p),
                            e_model_pred_dir=Path(a.cross_freq_e),
                            cross_proc_pred_dir=Path(a.cross_proc))
        if d is None:
            continue
        (t_mat, e_mat, _proxy, valid, min_len, m_t, e_m_t, cp_t, full_t) = d
        if full_t is None:
            continue
        n = min(min_len, a.max_chunks)
        t_mat, e_mat = t_mat[:n], e_mat[:n]
        full_t = full_t[:, :n, :]
        power = np.where(t_mat > 1e-12, e_mat / t_mat, 1.0)

        # --- 1. prediction bias, per (source,target) ---
        for si, src in enumerate(ALL_MODEL_CONFIGS):
            for ti, tgt in enumerate(CONFIGS):
                if src == tgt:
                    continue
                pred = full_t[si, :, ti]
                true = t_mat[:, ti]
                m = np.isfinite(pred) & (pred < 1e5) & (true > 0) & (true < 1e5)
                if m.sum() < 10:
                    continue
                bias[(src, tgt)].append(np.median((pred[m] - true[m]) / true[m]))

        # --- oracle greedy (true row i) ---
        true_score = edp(t_mat, e_mat, mp)
        oracle_pick = np.argmin(np.where(np.isfinite(true_score), true_score, np.inf), axis=1)

        # --- model greedy (reactive: row i-1 of the source-keyed tensor) ---
        start = CONFIGS.index('P_4.0GHz')
        model_pick = greedy_path(full_t, t_mat, e_mat, power, mp, start)

        for i in range(n):
            o, mo = int(oracle_pick[i]), int(model_pick[i])
            confusion[(CONFIGS[o], CONFIGS[mo])] += 1
            picked[CONFIGS[mo]] += 1
            regret[CONFIGS[mo]] += float(true_score[i, mo] - true_score[i, o])
            if i:
                trans[(CONFIGS[int(model_pick[i - 1])], CONFIGS[mo])] += 1
        n_chunks_tot += n

        # --- forecast tensor, if supplied ---
        if a.forecast_p and a.forecast_cp:
            f_p = _load_model_time_mat(wl, ph, Path(a.forecast_p), CONFIGS, n, t_mat, power)
            f_cp = _load_cross_proc_time_mat(wl, ph, Path(a.forecast_cp), CONFIGS, n, t_mat)
            f_e = (_load_e_model_time_mat(wl, ph, Path(a.forecast_e), CONFIGS, n, t_mat, power)
                   if a.forecast_e else None)
            f_full = _load_full_model_time_mat(f_p, f_cp, f_e)
            fc_pick = greedy_path(f_full, t_mat, e_mat, power, mp, start)
            for i in range(n):
                fc_confusion[(CONFIGS[int(oracle_pick[i])], CONFIGS[int(fc_pick[i])])] += 1
                fc_picked[CONFIGS[int(fc_pick[i])]] += 1

    # ================= report =================
    print('=' * 78)
    print('1. PREDICTION BIAS  (median signed rel. error of predicted time)')
    print('   negative => model thinks the target is FASTER than it is => over-selects it')
    print('=' * 78)
    print(f'{"source":10s} ' + ' '.join(f'{c:>9s}' for c in CONFIGS))
    for src in ALL_MODEL_CONFIGS:
        row = []
        for tgt in CONFIGS:
            v = bias.get((src, tgt))
            row.append('        .' if not v else f'{np.mean(v) * 100:+8.1f}%')
        print(f'{src:10s} ' + ' '.join(row))

    print()
    print('=' * 78)
    print('2. DECISION CONFUSION  (rows = oracle choice, cols = model choice, % of chunks)')
    print('=' * 78)
    tot = max(n_chunks_tot, 1)
    orows = sorted({o for o, _ in confusion}, key=CONFIGS.index)
    print(f'{"oracle\\model":14s} ' + ' '.join(f'{c:>9s}' for c in CONFIGS) + '   oracle%')
    for o in orows:
        cells = [confusion.get((o, m), 0) for m in CONFIGS]
        osum = sum(cells)
        print(f'{o:14s} ' + ' '.join(f'{c / tot * 100:8.1f}%' for c in cells)
              + f'  {osum / tot * 100:7.1f}%')
    agree = sum(v for (o, m), v in confusion.items() if o == m)
    print(f'\n   agreement: {agree / tot * 100:.1f}% of chunks')

    print()
    print('=' * 78)
    print('3. REGRET BY CHOSEN CONFIG  (excess true EDP caused by picking it)')
    print('=' * 78)
    tot_regret = sum(regret.values()) or 1.0
    print(f'{"model picked":12s} {"chunks%":>9s} {"share of regret":>16s}')
    for c in sorted(regret, key=lambda k: -regret[k]):
        print(f'{c:12s} {picked[c] / tot * 100:8.1f}% {regret[c] / tot_regret * 100:15.1f}%')

    print()
    print('=' * 78)
    print('4. ABSORBING BEHAVIOUR  (model policy config-to-config transition rates)')
    print('=' * 78)
    for frm in CONFIGS:
        out = {t: n for (f, t), n in trans.items() if f == frm}
        s = sum(out.values())
        if not s:
            continue
        stay = out.get(frm, 0) / s * 100
        leave = sorted(((n / s * 100, t) for t, n in out.items() if t != frm), reverse=True)[:2]
        esc = ', '.join(f'{t} {p:.1f}%' for p, t in leave) or 'never leaves'
        print(f'{frm:10s} stay={stay:5.1f}%  escapes: {esc}')

    if fc_picked:
        print()
        print('=' * 78)
        print('5. FORECAST vs REACTIVE  (config selection share)')
        print('=' * 78)
        print(f'{"config":10s} {"oracle%":>9s} {"reactive%":>10s} {"forecast%":>10s}')
        oshare = Counter()
        for (o, _m), v in confusion.items():
            oshare[o] += v
        for c in CONFIGS:
            print(f'{c:10s} {oshare[c] / tot * 100:8.1f}% {picked[c] / tot * 100:9.1f}%'
                  f' {fc_picked[c] / tot * 100:9.1f}%')
        fagree = sum(v for (o, m), v in fc_confusion.items() if o == m)
        print(f'\n   forecast agreement with oracle: {fagree / tot * 100:.1f}%'
              f'   (reactive: {agree / tot * 100:.1f}%)')


if __name__ == '__main__':
    main()

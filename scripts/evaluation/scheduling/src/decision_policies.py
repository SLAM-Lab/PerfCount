# decision_policies.py
#
# Generic, transition-aware decision-making functions (the DECISION-MAKING
# axis of the policy grid), plus a factory that wires a temporal window
# (temporal_policies.py) + a decision function into the
# `policy(time_mat, energy_mat, proxy_signal, configs, valid_configs,
# trans_lat, trans_nrg, metric) -> trace` shape used throughout main.py.
import numpy as np

import temporal_policies as temporal


def _exp(metric):
    return 2 if metric == 'ED2P' else 1


def decide_greedy(window_t, window_e, sub_lat, sub_nrg, prev_idx, metric):
    """Single-chunk window, transition-aware argmin.

    window_t/window_e: shape (1, n_actions) - the one chunk visible (the
    most recent past chunk for Reactive, or the current chunk for Oracle).
    prev_idx: previous action's index, or None to force zero transition cost
    (used at i==0).
    """
    p = _exp(metric)
    wt, we = window_t[0, :], window_e[0, :]
    if prev_idx is None:
        lat_costs, nrg_costs = wt, we
    else:
        lat_costs = sub_lat[prev_idx, :] + wt
        nrg_costs = sub_nrg[prev_idx, :] + we
    costs = nrg_costs * (lat_costs ** p)
    return int(np.argmin(costs))


def decide_mpc(window_t, window_e, sub_lat, sub_nrg, prev_idx, metric):
    """Multi-chunk window, Viterbi/DP over the window; returns first action
    of the optimal path.

    window_t/window_e: shape (L, n_actions).
    prev_idx: previous action's index, or None to force zero transition cost
    on the first window step (used at i==0).
    """
    p = _exp(metric)
    window_len, n_actions = window_t.shape

    dp_m = np.zeros((window_len, n_actions))
    parent_mat = np.zeros((window_len, n_actions), dtype=int)

    if prev_idx is None:
        lat_costs_0, nrg_costs_0 = window_t[0, :], window_e[0, :]
    else:
        lat_costs_0 = sub_lat[prev_idx, :] + window_t[0, :]
        nrg_costs_0 = sub_nrg[prev_idx, :] + window_e[0, :]
    dp_m[0, :] = nrg_costs_0 * (lat_costs_0 ** p)

    idx_arr = np.arange(n_actions)
    for w in range(1, window_len):
        lat_costs = sub_lat + window_t[w, :]
        nrg_costs = sub_nrg + window_e[w, :]
        step_metrics = nrg_costs * (lat_costs ** p)
        vals = dp_m[w - 1, :][:, None] + step_metrics
        best_prev = np.argmin(vals, axis=0)
        dp_m[w, :] = vals[best_prev, idx_arr]
        parent_mat[w, :] = best_prev

    curr = int(np.argmin(dp_m[window_len - 1, :]))
    for w in range(window_len - 1, 0, -1):
        curr = parent_mat[w, curr]
    return curr


def decide_global(time_mat_sub, energy_mat_sub, sub_lat, sub_nrg, metric):
    """Full-trace Viterbi DP. Returns the full action-index path."""
    p = _exp(metric)
    n_chunks, n_actions = time_mat_sub.shape

    dp_m = np.zeros((n_chunks, n_actions))
    parent_mat = np.zeros((n_chunks, n_actions), dtype=int)
    dp_m[0, :] = energy_mat_sub[0, :] * (time_mat_sub[0, :] ** p)
    idx_arr = np.arange(n_actions)

    for i in range(1, n_chunks):
        lat_costs = sub_lat + time_mat_sub[i, :]
        nrg_costs = sub_nrg + energy_mat_sub[i, :]
        step_metrics = nrg_costs * (lat_costs ** p)
        vals = dp_m[i - 1, :][:, None] + step_metrics
        best_prev = np.argmin(vals, axis=0)
        dp_m[i, :] = vals[best_prev, idx_arr]
        parent_mat[i, :] = best_prev

    best_idx = int(np.argmin(dp_m[-1, :]))
    path = np.zeros(n_chunks, dtype=int)
    curr = best_idx
    path[-1] = curr
    for i in range(n_chunks - 1, 0, -1):
        curr = parent_mat[i, curr]
        path[i - 1] = curr
    return path


def accumulate_trace(time_mat_sub, energy_mat_sub, sub_lat, sub_nrg, actions, metric):
    """Vectorized cumulative EDP/ED2P trace with transition costs.

    Transition cost at i=0 is forced to zero; all subsequent steps pay
    sub_lat[actions[i-1], actions[i]] / sub_nrg[...] for the config switch.
    """
    p = _exp(metric)
    acts = np.asarray(actions)
    n = len(acts)
    rows = np.arange(n)
    prev = np.empty(n, dtype=acts.dtype)
    prev[0] = acts[0]   # zeroed below; value doesn't matter
    prev[1:] = acts[:-1]
    lats = sub_lat[prev, acts].copy()
    nrgs = sub_nrg[prev, acts].copy()
    lats[0] = 0.0
    nrgs[0] = 0.0
    step_t = time_mat_sub[rows, acts] + lats
    step_e = energy_mat_sub[rows, acts] + nrgs
    return np.cumsum(step_e * (step_t ** p))


def compute_trace_stats(actions, local_names):
    """Compute per-policy migration and config-selection statistics.

    Returns a dict with:
      n_proc_migrations  — P↔E core-type switches
      n_freq_changes     — same core type, different frequency
      n_transitions      — total config changes (proc + freq)
      frac_<cfg>         — fraction of chunks spent on each config
      n_chunks           — total chunk count
    """
    n = len(actions)
    n_proc = 0
    n_freq = 0
    counts = {}
    for i in range(n):
        cfg = local_names[actions[i]]
        counts[cfg] = counts.get(cfg, 0) + 1
        if i == 0:
            continue
        prev = local_names[actions[i - 1]]
        curr = local_names[actions[i]]
        if prev == curr:
            continue
        if prev.split('_')[0] != curr.split('_')[0]:
            n_proc += 1
        else:
            n_freq += 1
    stats = {
        'n_proc_migrations': n_proc,
        'n_freq_changes': n_freq,
        'n_transitions': n_proc + n_freq,
        'n_chunks': n,
    }
    for name, cnt in counts.items():
        stats[f'frac_{name}'] = cnt / n if n > 0 else 0.0
    return stats


P_MODEL_FREQS = [1.0, 2.0, 3.0, 4.0]

# Source-config ordering for the cross-proc time tensor (axis 0).
ALL_MODEL_CONFIGS = [
    'E_1.0GHz', 'E_2.0GHz', 'E_3.0GHz', 'E_4.0GHz',
    'P_1.0GHz', 'P_2.0GHz', 'P_3.0GHz', 'P_4.0GHz',
]


def _src_freq_idx(configs, idx_list, local_prev_idx):
    """Map local action index to P_MODEL_FREQS axis-0 index (0-3).

    Used for P-core DVFS model_time_mat (4 source freqs). Returns 0 for
    any config not in P_MODEL_FREQS (e.g. E-core or P_5.0GHz).
    """
    global_cfg = configs[idx_list[local_prev_idx]]
    core, freq_str = global_cfg.split('_', 1)
    if core != 'P':
        return 0
    freq = float(freq_str.replace('GHz', ''))
    if freq not in P_MODEL_FREQS:
        return 0
    return P_MODEL_FREQS.index(freq)


def _src_cfg_idx(configs, idx_list, local_prev_idx):
    """Map local action index to ALL_MODEL_CONFIGS axis-0 index (0-7).

    Used for cross-proc cross_proc_time_mat (8 source configs: E_1-4, P_1-4).
    Returns 0 for any config not in ALL_MODEL_CONFIGS.
    """
    global_cfg = configs[idx_list[local_prev_idx]]
    if global_cfg not in ALL_MODEL_CONFIGS:
        return 0
    return ALL_MODEL_CONFIGS.index(global_cfg)


def decide_model_global(model_sub_t, model_sub_e, sub_lat, sub_nrg, metric):
    """Full-trace Viterbi where step cost at chunk i depends on the previous state.

    model_sub_t/model_sub_e: shape (n_src, n_chunks, n_actions)
      model_sub_t[a, i, b] = predicted time for config b at chunk i when the
      scheduler was last running config a.  For P-core DVFS: n_src == n_actions.

    Returns the full action-index path (local indices).
    """
    p = _exp(metric)
    n_src, n_chunks, n_actions = model_sub_t.shape

    dp_m = np.zeros((n_chunks, n_actions))
    parent_mat = np.zeros((n_chunks, n_actions), dtype=int)

    # i=0: no previous state; use diagonal (oracle ground-truth) as initial cost
    for b in range(n_actions):
        src_for_b = min(b, n_src - 1)
        dp_m[0, b] = model_sub_e[src_for_b, 0, b] * (model_sub_t[src_for_b, 0, b] ** p)

    idx_arr = np.arange(n_actions)
    for i in range(1, n_chunks):
        # step_cost[a, b] = (nrg_trans[a,b] + model_e[a,i,b]) * (lat_trans[a,b] + model_t[a,i,b])^p
        # model_sub_t[:, i, :] has shape (n_src, n_actions); if n_src==n_actions, index by a directly
        model_t_i = model_sub_t[:n_actions, i, :]   # (n_actions, n_actions)
        model_e_i = model_sub_e[:n_actions, i, :]   # (n_actions, n_actions)
        lat_total = sub_lat + model_t_i              # (n_actions, n_actions)
        nrg_total = sub_nrg + model_e_i              # (n_actions, n_actions)
        step_cost = nrg_total * (lat_total ** p)     # (n_actions, n_actions)

        vals = dp_m[i - 1, :][:, None] + step_cost  # (n_actions, n_actions)
        best_prev = np.argmin(vals, axis=0)
        dp_m[i, :] = vals[best_prev, idx_arr]
        parent_mat[i, :] = best_prev

    best_idx = int(np.argmin(dp_m[-1, :]))
    path = np.zeros(n_chunks, dtype=int)
    curr = best_idx
    path[-1] = curr
    for i in range(n_chunks - 1, 0, -1):
        curr = parent_mat[i, curr]
        path[i - 1] = curr
    return path


def _dampen_predictions(model_sub_t, model_sub_e, si, row, n_configs,
                        history_t, dampen_window):
    """Blend model predictions toward the rolling mean when variance is high.

    Tracks a rolling window of recent per-config predicted times. When the
    coefficient of variation (std/mean) for any config exceeds a threshold,
    blends that config's prediction toward the window mean proportionally to
    the CV. This prevents the scheduler from reacting to volatile predictions.

    Returns dampened (time_row, energy_row) as (1, n_configs) arrays.
    """
    raw_t = model_sub_t[si, row, :]
    raw_e = model_sub_e[si, row, :]

    history_t.append(raw_t.copy())
    if len(history_t) > dampen_window:
        history_t.pop(0)

    if len(history_t) < 3:
        return raw_t[np.newaxis, :], raw_e[np.newaxis, :]

    hist = np.array(history_t)
    mu = hist.mean(axis=0)
    std = hist.std(axis=0)
    cv = std / (mu + 1e-12)

    # Blend factor: 0 when cv < 0.05 (stable), ramps to 1 when cv >= 0.3
    alpha = np.clip((cv - 0.05) / 0.25, 0.0, 1.0)

    dampened_t = (1.0 - alpha) * raw_t + alpha * mu
    dampened_e = (1.0 - alpha) * raw_e + alpha * (raw_e / (raw_t + 1e-12)) * dampened_t

    return dampened_t[np.newaxis, :], dampened_e[np.newaxis, :]


def make_model_policy_from_idx_list(idx_list_fn, decision_mode,
                                     window_size=None, start_idx_fn=None,
                                     src_idx_fn=None, temporal='reactive',
                                     lookahead_k=0, dampen_window=0):
    """Factory for model-based policies.

    Returned policy signature:
        policy(time_mat, energy_mat, proxy_signal, configs, valid_configs,
               trans_lat, trans_nrg, model_time_mat, metric) -> trace

    model_time_mat: ndarray (n_src, n_chunks, n_configs) — precomputed model
        predictions.  model_time_mat[si, i, ci] is the predicted time for config
        ci at chunk i when the scheduler was last on the source config at axis-0
        index si.

    src_idx_fn: maps (configs, idx_list, local_prev_idx) -> axis-0 index into
        model_time_mat.  Defaults to _src_freq_idx (P-core DVFS, 4 sources).
        Pass _src_cfg_idx for cross-proc policies (8 sources: E_1-4, P_1-4).

    temporal: 'reactive' (default) uses chunk i-1's model row (prior PMU data,
        causal); 'oracle' uses chunk i's model row (perfect-future knowledge).

    dampen_window: if >0, enables rolling-window variance dampening. The model's
        predicted times are tracked over the last `dampen_window` chunks; when the
        coefficient of variation is high (volatile predictions), the prediction is
        blended toward the rolling mean to avoid reacting to outlier predictions.
        This is implementable at runtime with zero extra inference cost.
    """
    _src_fn = src_idx_fn if src_idx_fn is not None else _src_freq_idx

    def policy(time_mat, energy_mat, proxy_signal, configs, valid_configs,
               trans_lat, trans_nrg, model_time_mat, metric, _return_actions=False):
        n_chunks = len(time_mat)
        idx_list = idx_list_fn(configs, valid_configs)
        if not idx_list:
            return (np.zeros(n_chunks), np.zeros(n_chunks, dtype=int), []) if _return_actions else np.zeros(n_chunks)

        sub_lat = trans_lat[np.ix_(idx_list, idx_list)]
        sub_nrg = trans_nrg[np.ix_(idx_list, idx_list)]
        local_names = [configs[ci] for ci in idx_list]

        model_sub_t = model_time_mat[:, :n_chunks, :][:, :, idx_list]
        oracle_power = np.where(time_mat[:, idx_list] > 1e-12,
                                energy_mat[:, idx_list] / time_mat[:, idx_list],
                                1.0)
        model_sub_e = model_sub_t * oracle_power[np.newaxis, :, :]

        if decision_mode == 'global':
            path = decide_model_global(model_sub_t, model_sub_e, sub_lat, sub_nrg, metric)
            oracle_sub_t = time_mat[:, idx_list]
            oracle_sub_e = energy_mat[:, idx_list]
            trace = accumulate_trace(oracle_sub_t, oracle_sub_e, sub_lat, sub_nrg, path, metric)
            return (trace, path, local_names) if _return_actions else trace

        start_idx = start_idx_fn(idx_list, valid_configs) if start_idx_fn else len(idx_list) - 1
        actions = np.zeros(n_chunks, dtype=int)
        prev_idx = start_idx
        history_t = []

        for i in range(n_chunks):
            si = _src_fn(configs, idx_list, prev_idx)
            pidx = prev_idx if i > 0 else None

            if i == 0:
                action = start_idx
            elif decision_mode == 'greedy':
                row = i if temporal == 'oracle' else i - 1
                if dampen_window > 0:
                    window_t, window_e = _dampen_predictions(
                        model_sub_t, model_sub_e, si, row,
                        len(idx_list), history_t, dampen_window)
                elif lookahead_k > 0 and temporal == 'oracle':
                    end = min(i + lookahead_k + 1, n_chunks)
                    window_t = model_sub_t[si, i:end, :].mean(axis=0, keepdims=True)
                    window_e = model_sub_e[si, i:end, :].mean(axis=0, keepdims=True)
                else:
                    window_t = model_sub_t[si, row, :][np.newaxis, :]
                    window_e = model_sub_e[si, row, :][np.newaxis, :]
                action = decide_greedy(window_t, window_e, sub_lat, sub_nrg, pidx, metric)
            elif decision_mode == 'mpc':
                W = window_size or 1
                n_future = min(W, n_chunks - i)
                if temporal == 'oracle':
                    window_t = model_sub_t[si, i:i + n_future, :]
                    window_e = model_sub_e[si, i:i + n_future, :]
                else:
                    row = i - 1
                    window_t = np.tile(model_sub_t[si, row, :], (n_future, 1))
                    window_e = np.tile(model_sub_e[si, row, :], (n_future, 1))
                if n_future == 1:
                    action = decide_greedy(window_t, window_e, sub_lat, sub_nrg, pidx, metric)
                else:
                    action = decide_mpc(window_t, window_e, sub_lat, sub_nrg, pidx, metric)
            else:
                raise ValueError(f"Unknown decision_mode for model policy: {decision_mode}")

            actions[i] = action
            prev_idx = action

        oracle_sub_t = time_mat[:, idx_list]
        oracle_sub_e = energy_mat[:, idx_list]
        trace = accumulate_trace(oracle_sub_t, oracle_sub_e, sub_lat, sub_nrg, actions, metric)
        return (trace, actions, local_names) if _return_actions else trace

    policy.metric_independent = False
    policy.returns_actions = True
    policy.is_viterbi_oracle = (decision_mode == 'global')
    return policy


def make_policy_from_idx_list(idx_list_fn, temporal_mode, decision_mode,
                               window_size=None, start_idx_fn=None,
                               heuristic_fn=None, initial_state=None,
                               metric_independent=False, batch_decide_fn=None):
    """Factory returning a `policy(...) -> trace` closure.

    idx_list_fn(configs, valid_configs) -> list[int]
        Restricted action-index list (columns of time_mat/energy_mat/
        trans_lat/trans_nrg this policy is allowed to choose among).
    temporal_mode: 'reactive' | 'oracle'
    decision_mode: 'greedy' | 'mpc' | 'global' | 'heuristic'
    window_size: required for greedy (=1) / mpc (=horizon) / heuristic
        (governor's counter window); ignored for global.
    start_idx_fn(idx_list, valid_configs) -> int
        Initial action index (local to idx_list) for greedy/mpc/heuristic.
        Defaults to len(idx_list) - 1 (max-frequency / last entry).
    heuristic_fn, initial_state: only for decision_mode='heuristic'.
        heuristic_fn(ctx, state) -> (action_idx, new_state); initial_state
        may be a dict (copied per call) or a zero-arg callable returning a
        fresh dict per call (needed for e.g. per-call RNG state).
    metric_independent: if True, heuristic_fn does not read ctx['metric'].
        Enables dual-metric fast path: pass metrics=['EDP','ED2P'] to get
        {metric: trace} dict from a single decision-loop pass.
    """
    def policy(time_mat, energy_mat, proxy_signal, configs, valid_configs, trans_lat, trans_nrg, metric='EDP',
               _return_actions=False, metrics=None):
        n_chunks = len(time_mat)
        idx_list = idx_list_fn(configs, valid_configs)
        if not idx_list:
            return (np.zeros(n_chunks), np.zeros(n_chunks, dtype=int), []) if _return_actions else np.zeros(n_chunks)

        sub_t = time_mat[:, idx_list]
        sub_e = energy_mat[:, idx_list]
        sub_lat = trans_lat[np.ix_(idx_list, idx_list)]
        sub_nrg = trans_nrg[np.ix_(idx_list, idx_list)]
        local_names = [configs[ci] for ci in idx_list]

        if decision_mode == 'global':
            path = decide_global(sub_t, sub_e, sub_lat, sub_nrg, metric)
            trace = accumulate_trace(sub_t, sub_e, sub_lat, sub_nrg, path, metric)
            return (trace, path, local_names) if _return_actions else trace

        start_idx = start_idx_fn(idx_list, valid_configs) if start_idx_fn else len(idx_list) - 1

        if initial_state is None:
            state = {}
        elif callable(initial_state):
            state = initial_state()
        else:
            state = dict(initial_state)

        actions = np.zeros(n_chunks, dtype=int)
        prev_idx = start_idx

        if batch_decide_fn is not None:
            actions = batch_decide_fn(
                proxy_signal, sub_t, sub_e, sub_lat, sub_nrg,
                n_chunks, start_idx, valid_configs, idx_list,
            )
        else:
            for i in range(n_chunks):
                pidx = prev_idx if i > 0 else None

                if decision_mode == 'heuristic':
                    if temporal_mode == 'oracle_heuristic':
                        # Perfect-future heuristic: sees current chunk i's true data
                        prev_t = sub_t[i, :]
                        prev_e = sub_e[i, :]
                        proxy_win = proxy_signal[i:i+1]
                    else:
                        prev_t = sub_t[i - 1, :] if i > 0 else None
                        prev_e = sub_e[i - 1, :] if i > 0 else None
                        proxy_win = temporal.reactive_signal_window(proxy_signal, i, window_size)
                    ctx = {
                        'i': i,
                        'metric': metric,
                        'proxy_window': proxy_win,
                        'prev_t': prev_t,
                        'prev_e': prev_e,
                        'sub_t': sub_t,
                        'sub_e': sub_e,
                        'sub_lat': sub_lat,
                        'sub_nrg': sub_nrg,
                        'prev_idx': pidx,
                        'start_idx': start_idx,
                        'valid_configs': valid_configs,
                    }
                    action, state = heuristic_fn(ctx, state)
                else:
                    if temporal_mode == 'reactive':
                        window_t, window_e = temporal.reactive_window_raw(sub_t, sub_e, i, window_size)
                    elif temporal_mode == 'reactive_oracle':
                        window_t, window_e = temporal.reactive_oracle_window_raw(sub_t, sub_e, i)
                    else:  # 'oracle' — perfect future: sees chunk i's true data
                        window_t, window_e = temporal.oracle_window_raw(sub_t, sub_e, i, window_size)

                    if window_t.shape[0] == 0:
                        action = start_idx
                    elif decision_mode == 'greedy':
                        action = decide_greedy(window_t, window_e, sub_lat, sub_nrg, pidx, metric)
                    elif decision_mode == 'mpc':
                        action = decide_mpc(window_t, window_e, sub_lat, sub_nrg, pidx, metric)
                    else:
                        raise ValueError(f"Unknown decision_mode: {decision_mode}")

                actions[i] = action
                prev_idx = action

        if metrics is not None:
            return {m: accumulate_trace(sub_t, sub_e, sub_lat, sub_nrg, actions, m)
                    for m in metrics}
        trace = accumulate_trace(sub_t, sub_e, sub_lat, sub_nrg, actions, metric)
        return (trace, actions, local_names) if _return_actions else trace

    policy.metric_independent = metric_independent
    policy.returns_actions = True
    policy.is_viterbi_oracle = (decision_mode == 'global')
    return policy

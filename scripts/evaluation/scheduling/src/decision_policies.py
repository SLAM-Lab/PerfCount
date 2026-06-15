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
    """Replay loop: given chosen `actions[i]` per chunk, applies transition
    costs between actions[i-1]/actions[i] (zero at i==0), and computes the
    cumulative EDP/ED2P trace.
    """
    p = _exp(metric)
    n_chunks = len(actions)
    trace = np.zeros(n_chunks)
    cum_m = 0.0
    for i in range(n_chunks):
        a = actions[i]
        if i == 0:
            lat, nrg = 0.0, 0.0
        else:
            prev = actions[i - 1]
            lat, nrg = sub_lat[prev, a], sub_nrg[prev, a]
        step_t = lat + time_mat_sub[i, a]
        step_e = nrg + energy_mat_sub[i, a]
        cum_m += step_e * (step_t ** p)
        trace[i] = cum_m
    return trace


def make_policy_from_idx_list(idx_list_fn, temporal_mode, decision_mode,
                               window_size=None, start_idx_fn=None,
                               heuristic_fn=None, initial_state=None):
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
    """
    def policy(time_mat, energy_mat, proxy_signal, configs, valid_configs, trans_lat, trans_nrg, metric):
        n_chunks = len(time_mat)
        idx_list = idx_list_fn(configs, valid_configs)
        if not idx_list:
            return np.zeros(n_chunks)

        sub_t = time_mat[:, idx_list]
        sub_e = energy_mat[:, idx_list]
        sub_lat = trans_lat[np.ix_(idx_list, idx_list)]
        sub_nrg = trans_nrg[np.ix_(idx_list, idx_list)]

        if decision_mode == 'global':
            path = decide_global(sub_t, sub_e, sub_lat, sub_nrg, metric)
            return accumulate_trace(sub_t, sub_e, sub_lat, sub_nrg, path, metric)

        start_idx = start_idx_fn(idx_list, valid_configs) if start_idx_fn else len(idx_list) - 1

        if initial_state is None:
            state = {}
        elif callable(initial_state):
            state = initial_state()
        else:
            state = dict(initial_state)

        actions = np.zeros(n_chunks, dtype=int)
        prev_idx = start_idx

        for i in range(n_chunks):
            pidx = prev_idx if i > 0 else None

            if decision_mode == 'heuristic':
                prev_t, prev_e = (sub_t[i - 1, :], sub_e[i - 1, :]) if i > 0 else (None, None)
                ctx = {
                    'i': i,
                    'metric': metric,
                    'proxy_window': temporal.reactive_signal_window(proxy_signal, i, window_size),
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
                else:
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

        return accumulate_trace(sub_t, sub_e, sub_lat, sub_nrg, actions, metric)

    return policy

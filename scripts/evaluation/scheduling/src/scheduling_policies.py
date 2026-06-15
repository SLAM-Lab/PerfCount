# scheduling_policies.py
#
# Heterogeneous (P+E) core-placement + DVFS scheduling policies. Oracle/
# Global policies and SOTA/heuristic governors are thin wrappers around
# decision_policies.make_policy_from_idx_list.
import numpy as np

from decision_policies import make_policy_from_idx_list


def _full_idx_list_fn(configs, valid_configs):
    return [configs.index(c) for c in valid_configs]


def _last_core_local_idx(valid_configs, core_type):
    """Local index (within the full valid_configs ordering) of the last
    (max-frequency) config of the given core type, or 0 if none exists."""
    positions = [i for i, c in enumerate(valid_configs) if c.startswith(core_type)]
    return positions[-1] if positions else 0


def _p_max_start_idx(idx_list, valid_configs):
    return _last_core_local_idx(valid_configs, 'P')


# ==========================================
# GLOBAL HETERO ORACLE (Global, Oracle, full P+E x freq grid)
# ==========================================
def run_proactive_hetero_oracle(time_mat, energy_mat, proxy_signal, configs, valid_configs, trans_lat, trans_nrg, metric):
    """Viterbi Dynamic Programming for Global Minimum across the full P+E x freq grid."""
    policy = make_policy_from_idx_list(
        idx_list_fn=_full_idx_list_fn,
        temporal_mode='oracle',
        decision_mode='global',
    )
    return policy(time_mat, energy_mat, proxy_signal, configs, valid_configs, trans_lat, trans_nrg, metric)


# ==========================================
# MICRO EAS (Heuristic, Reactive, simplified 2-config P<->E)
# ==========================================
def _decide_heuristic_micro_eas(ctx, state):
    if 'p_idx' not in state:
        valid = ctx['valid_configs']
        state = dict(state)
        state['p_idx'] = [k for k, c in enumerate(valid) if c.startswith('P')]
        state['e_idx'] = [k for k, c in enumerate(valid) if c.startswith('E')]
        state['e_mid'] = state['e_idx'][len(state['e_idx']) // 2]
        state['p_max'] = state['p_idx'][-1]
        state['mig_nrg_cost'] = ctx['sub_nrg'][state['p_max'], state['e_mid']]

    p_idx, e_mid, p_max, mig_nrg_cost = state['p_idx'], state['e_mid'], state['p_max'], state['mig_nrg_cost']

    curr = ctx['start_idx'] if ctx['i'] == 0 else ctx['prev_idx']
    if ctx['i'] > 0:
        if curr in p_idx:
            e_saving = ctx['prev_e'][curr] - ctx['prev_e'][e_mid]
            if e_saving > mig_nrg_cost:
                curr = e_mid
        else:
            if ctx['proxy_window'][-1] > 2.5:
                curr = p_max
    return curr, state


def run_micro_eas(time_mat, energy_mat, proxy_signal, configs, valid_configs, trans_lat, trans_nrg, metric):
    policy = make_policy_from_idx_list(
        idx_list_fn=_full_idx_list_fn,
        temporal_mode='reactive',
        decision_mode='heuristic',
        window_size=1,
        start_idx_fn=_p_max_start_idx,
        heuristic_fn=_decide_heuristic_micro_eas,
    )
    return policy(time_mat, energy_mat, proxy_signal, configs, valid_configs, trans_lat, trans_nrg, metric)


# ==========================================
# REACTIVE / PROACTIVE N-STEP, FIXED FREQUENCY (IsoFreq ablations)
# ==========================================
def _fixed_freq_idx_list_fn(target_freq):
    def idx_list_fn(configs, valid_configs):
        return [configs.index(c) for c in valid_configs if target_freq in c]
    return idx_list_fn


def _fixed_freq_start_idx_fn(target_freq):
    def start_idx_fn(idx_list, valid_configs):
        freq_configs = [c for c in valid_configs if target_freq in c]
        p_positions = [i for i, c in enumerate(freq_configs) if c.startswith('P')]
        return p_positions[0] if p_positions else 0
    return start_idx_fn


def make_reactive_n_step_fixed_freq(lookback, target_freq):
    """Reactive History Lookback locked to a specific frequency."""
    return make_policy_from_idx_list(
        idx_list_fn=_fixed_freq_idx_list_fn(target_freq),
        temporal_mode='reactive',
        decision_mode='mpc',
        window_size=lookback,
        start_idx_fn=_fixed_freq_start_idx_fn(target_freq),
    )


def make_proactive_n_step_fixed_freq(horizon, target_freq):
    """Proactive MPC Lookahead locked to a specific frequency."""
    return make_policy_from_idx_list(
        idx_list_fn=_fixed_freq_idx_list_fn(target_freq),
        temporal_mode='oracle',
        decision_mode='mpc',
        window_size=horizon,
        start_idx_fn=_fixed_freq_start_idx_fn(target_freq),
    )


def make_global_oracle_fixed_freq(target_freq):
    """Global Viterbi DP locked to a specific frequency."""
    return make_policy_from_idx_list(
        idx_list_fn=_fixed_freq_idx_list_fn(target_freq),
        temporal_mode='oracle',
        decision_mode='global',
    )


# ==========================================
# LINUX EAS (Energy Aware Scheduling) (Heuristic, Reactive)
# ==========================================
def make_eas_hetero(alpha=0.25, perf_slack=0.10):
    """
    Linux Energy Aware Scheduling (EAS).
    At each interval: among all configs where previous-chunk time <= P_max_time * (1+perf_slack),
    select the minimum-energy configuration. Falls back to P-core if no config qualifies.
    PELT-style EWMA smooths utilization for scheduling stability.
    Ref: Quentin Perret et al., "An Energy Model for the Linux Kernel" (ELC 2018).
    """
    def decide(ctx, state):
        if 'p_idx' not in state:
            valid = ctx['valid_configs']
            state = dict(state)
            state['p_idx'] = sorted([k for k, c in enumerate(valid) if c.startswith('P')])
            state['e_idx'] = sorted([k for k, c in enumerate(valid) if c.startswith('E')])

        if ctx['i'] == 0:
            return ctx['start_idx'], state

        p_idx, e_idx = state['p_idx'], state['e_idx']
        n = ctx['sub_t'].shape[1]
        util_ewma = state.get('util_ewma', 1.0)
        raw_util = np.clip((ctx['proxy_window'][-1] - 1.0) / 2.5, 0.0, 1.0)
        util_ewma = alpha * raw_util + (1.0 - alpha) * util_ewma

        p_max = p_idx[-1]
        perf_limit = ctx['prev_t'][p_max] * (1.0 + perf_slack)
        feasible = [c for c in range(n) if ctx['prev_t'][c] <= perf_limit]
        if not feasible:
            feasible = p_idx

        action = min(feasible, key=lambda c: ctx['prev_e'][c])

        state = dict(state)
        state['util_ewma'] = util_ewma
        return action, state

    return make_policy_from_idx_list(
        idx_list_fn=_full_idx_list_fn,
        temporal_mode='reactive',
        decision_mode='heuristic',
        window_size=1,
        start_idx_fn=_p_max_start_idx,
        heuristic_fn=decide,
    )


# ==========================================
# THRESHOLD MIGRATION WITH HYSTERESIS (Heuristic, Reactive)
# ==========================================
def make_threshold_migration(up_thresh=0.60, down_thresh=0.30, alpha=0.25):
    """
    Hysteresis-based big.LITTLE cluster migration (ARM GTS-style).
    Maintains EWMA of proxy utilization. Migrates E->P when EWMA exceeds up_thresh;
    migrates P->E when EWMA falls below down_thresh. The hysteresis gap between
    thresholds prevents rapid oscillation at the boundary. Within each cluster,
    always uses maximum frequency (no per-cluster DVFS).
    Ref: ARM big.LITTLE Technology: The Future of Mobile (Greenhalgh 2011);
         ARM GTS (Global Task Scheduling) white paper.
    """
    def decide(ctx, state):
        if 'p_max' not in state:
            valid = ctx['valid_configs']
            state = dict(state)
            state['p_max'] = _last_core_local_idx(valid, 'P')
            state['e_max'] = _last_core_local_idx(valid, 'E')

        if ctx['i'] == 0:
            return ctx['start_idx'], state

        p_max, e_max = state['p_max'], state['e_max']
        on_p_core = state.get('on_p_core', True)
        util_ewma = state.get('util_ewma', 1.0)
        raw_util = np.clip((ctx['proxy_window'][-1] - 1.0) / 2.5, 0.0, 1.0)
        util_ewma = alpha * raw_util + (1.0 - alpha) * util_ewma

        if on_p_core and util_ewma < down_thresh:
            on_p_core = False
        elif not on_p_core and util_ewma > up_thresh:
            on_p_core = True

        action = p_max if on_p_core else e_max

        state = dict(state)
        state['on_p_core'] = on_p_core
        state['util_ewma'] = util_ewma
        return action, state

    return make_policy_from_idx_list(
        idx_list_fn=_full_idx_list_fn,
        temporal_mode='reactive',
        decision_mode='heuristic',
        window_size=1,
        start_idx_fn=_p_max_start_idx,
        heuristic_fn=decide,
    )


# ==========================================
# EAS + DVFS (Combined Modern Linux) (Heuristic, Reactive)
# ==========================================
def make_eas_with_dvfs(alpha=0.25, headroom=1.25, perf_slack=0.10):
    """
    Combined Linux EAS + schedutil (modern Linux behavior, v5.0+).
    Step 1 (EAS): checks if any E-core frequency meets performance constraint
    (prev-chunk time <= P_max * (1+perf_slack)); if yes, selects E-core cluster.
    Step 2 (schedutil): within the chosen cluster, PELT util * headroom selects
    the frequency level, matching Linux schedutil governor behavior.
    Ref: Linux kernel v5.0+ EAS + schedutil interaction documentation.
    """
    def decide(ctx, state):
        if 'p_idx' not in state:
            valid = ctx['valid_configs']
            state = dict(state)
            state['p_idx'] = sorted([k for k, c in enumerate(valid) if c.startswith('P')])
            state['e_idx'] = sorted([k for k, c in enumerate(valid) if c.startswith('E')])

        if ctx['i'] == 0:
            return ctx['start_idx'], state

        p_idx, e_idx = state['p_idx'], state['e_idx']
        util_ewma = state.get('util_ewma', 1.0)
        raw_util = np.clip((ctx['proxy_window'][-1] - 1.0) / 2.5, 0.0, 1.0)
        util_ewma = alpha * raw_util + (1.0 - alpha) * util_ewma
        target_ratio = min(util_ewma * headroom, 1.0)

        p_max = p_idx[-1]
        perf_limit = ctx['prev_t'][p_max] * (1.0 + perf_slack)
        e_feasible = sorted([c for c in e_idx if ctx['prev_t'][c] <= perf_limit])

        if e_feasible:
            level = int(np.round(target_ratio * (len(e_feasible) - 1)))
            action = e_feasible[np.clip(level, 0, len(e_feasible) - 1)]
        else:
            level = int(np.round(target_ratio * (len(p_idx) - 1)))
            action = p_idx[np.clip(level, 0, len(p_idx) - 1)]

        state = dict(state)
        state['util_ewma'] = util_ewma
        return action, state

    return make_policy_from_idx_list(
        idx_list_fn=_full_idx_list_fn,
        temporal_mode='reactive',
        decision_mode='heuristic',
        window_size=1,
        start_idx_fn=_p_max_start_idx,
        heuristic_fn=decide,
    )


# ==========================================
# INTEL THREAD DIRECTOR STYLE (Heuristic, Reactive)
# ==========================================
def make_thread_director(window=5, compute_thresh=0.60, alpha=0.25, headroom=1.25):
    """
    Intel Thread Director (ITD) classification-based heterogeneous scheduling.
    Classifies workload type via a sliding window average of the proxy signal:
    high average proxy -> compute-heavy (benefits from P-core);
    low average proxy -> memory-bound (E-core sufficient).
    The compute score is then used to select the cluster; within the chosen
    cluster, schedutil-style frequency scaling is applied via PELT EWMA.
    Ref: Intel Thread Director / Enhanced Hardware Feedback Interface (EHFI),
         Intel Alder Lake Architecture (2021).
    """
    def decide(ctx, state):
        if 'p_idx' not in state:
            valid = ctx['valid_configs']
            state = dict(state)
            state['p_idx'] = sorted([k for k, c in enumerate(valid) if c.startswith('P')])
            state['e_idx'] = sorted([k for k, c in enumerate(valid) if c.startswith('E')])

        if ctx['i'] == 0:
            return ctx['start_idx'], state

        p_idx, e_idx = state['p_idx'], state['e_idx']
        proxy_window = ctx['proxy_window']
        avg_proxy = np.mean(proxy_window)
        compute_score = np.clip((avg_proxy - 1.0) / 2.5, 0.0, 1.0)

        raw_util = np.clip((proxy_window[-1] - 1.0) / 2.5, 0.0, 1.0)
        util_ewma = state.get('util_ewma', 1.0)
        util_ewma = alpha * raw_util + (1.0 - alpha) * util_ewma
        target_ratio = min(util_ewma * headroom, 1.0)

        if compute_score >= compute_thresh:
            level = int(np.round(target_ratio * (len(p_idx) - 1)))
            action = p_idx[np.clip(level, 0, len(p_idx) - 1)]
        else:
            level = int(np.round(target_ratio * (len(e_idx) - 1)))
            action = e_idx[np.clip(level, 0, len(e_idx) - 1)]

        state = dict(state)
        state['util_ewma'] = util_ewma
        return action, state

    return make_policy_from_idx_list(
        idx_list_fn=_full_idx_list_fn,
        temporal_mode='reactive',
        decision_mode='heuristic',
        window_size=window,
        start_idx_fn=_p_max_start_idx,
        heuristic_fn=decide,
    )


# ==========================================
# UCB1 HETERO BANDIT (Full P+E Space) (Heuristic, Reactive)
# ==========================================
def make_ucb1_hetero(c=1.0):
    """
    UCB1 multi-armed bandit across the full heterogeneous configuration space (P+E x all freqs).
    Treats each (core_type, frequency) pair as a bandit arm. Rewards are the negative
    per-chunk EDP/ED2P cost, normalized online by the running maximum.
    Initialization: round-robins through all valid configs once before exploiting.
    Extends make_ucb1_dvfs (single-cluster) to the full P+E decision space.
    Ref: Auer et al., "Finite-time Analysis of the Multiarmed Bandit Problem" (2002).
    """
    def decide(ctx, state):
        n = ctx['sub_t'].shape[1]
        i = ctx['i']
        counts = state.get('counts', np.zeros(n))
        values = state.get('values', np.zeros(n))
        max_cost_seen = state.get('max_cost_seen', 1e-12)
        p = 2 if ctx['metric'] == 'ED2P' else 1

        if i < n:
            arm = i
        else:
            ucb = values + c * np.sqrt(2.0 * np.log(i) / counts)
            arm = int(np.argmax(ucb))

        prev_idx = ctx['prev_idx']
        if prev_idx is None:
            lat, nrg = 0.0, 0.0
        else:
            lat, nrg = ctx['sub_lat'][prev_idx, arm], ctx['sub_nrg'][prev_idx, arm]
        step_t = lat + ctx['sub_t'][i, arm]
        step_e = nrg + ctx['sub_e'][i, arm]
        step_m = step_e * (step_t ** p)

        max_cost_seen = max(max_cost_seen, step_m)
        reward = -step_m / max_cost_seen
        counts = counts.copy()
        values = values.copy()
        counts[arm] += 1
        values[arm] += (reward - values[arm]) / counts[arm]

        return arm, {'counts': counts, 'values': values, 'max_cost_seen': max_cost_seen}

    return make_policy_from_idx_list(
        idx_list_fn=_full_idx_list_fn,
        temporal_mode='reactive',
        decision_mode='heuristic',
        window_size=1,
        start_idx_fn=_p_max_start_idx,
        heuristic_fn=decide,
    )

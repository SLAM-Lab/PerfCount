# scheduling_policies.py
#
# Heterogeneous (P+E) core-placement + DVFS scheduling policies. Oracle/
# Global policies and SOTA/heuristic governors are thin wrappers around
# decision_policies.make_policy_from_idx_list.
import numpy as np

from decision_policies import (make_policy_from_idx_list,
                               make_model_policy_from_idx_list, _src_cfg_idx)
from data_loader import ALL_MODEL_CONFIGS


def _full_idx_list_fn(configs, valid_configs):
    return [configs.index(c) for c in valid_configs]


def _last_core_local_idx(valid_configs, core_type):
    """Local index (within the full valid_configs ordering) of the last
    (max-frequency) config of the given core type, or 0 if none exists."""
    positions = [i for i, c in enumerate(valid_configs) if c.startswith(core_type)]
    return positions[-1] if positions else 0


def _p_max_start_idx(idx_list, valid_configs):
    return _last_core_local_idx(valid_configs, 'P')


def _norm_proxy(ctx):
    return np.clip((ctx['proxy_window'][-1] - 1.0) / 2.5, 0.0, 1.0)

def _ewma_step(raw_util, prev_ewma, alpha):
    return alpha * raw_util + (1.0 - alpha) * prev_ewma

def _freq_select(idx_list, target_ratio):
    n = len(idx_list)
    level = int(np.clip(np.round(target_ratio * (n - 1)), 0, n - 1))
    return idx_list[level]

def _perf_feasible(ctx, perf_slack, p_max_local):
    limit = ctx['prev_t'][p_max_local] * (1.0 + perf_slack)
    n = ctx['sub_t'].shape[1]
    return [j for j in range(n) if ctx['prev_t'][j] <= limit]

def _ensure_pe_indices(state, ctx):
    if 'p_idx' not in state:
        valid = ctx['valid_configs']
        state = dict(state)
        state['p_idx'] = sorted([k for k, c in enumerate(valid) if c.startswith('P')])
        state['e_idx'] = sorted([k for k, c in enumerate(valid) if c.startswith('E')])
    return state


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

run_proactive_hetero_oracle.is_viterbi_oracle = True


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
        metric_independent=True,
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
        state = _ensure_pe_indices(state, ctx)
        if ctx['i'] == 0:
            return ctx['start_idx'], state
        p_idx = state['p_idx']
        util_ewma = _ewma_step(_norm_proxy(ctx), state.get('util_ewma', 1.0), alpha)
        feasible = _perf_feasible(ctx, perf_slack, p_idx[-1]) or p_idx
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
        metric_independent=True,
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
        state = _ensure_pe_indices(state, ctx)
        if ctx['i'] == 0:
            return ctx['start_idx'], state
        on_p_core = state.get('on_p_core', True)
        util_ewma = _ewma_step(_norm_proxy(ctx), state.get('util_ewma', 1.0), alpha)
        if on_p_core and util_ewma < down_thresh:
            on_p_core = False
        elif not on_p_core and util_ewma > up_thresh:
            on_p_core = True
        action = state['p_idx'][-1] if on_p_core else state['e_idx'][-1]
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
        metric_independent=True,
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
        state = _ensure_pe_indices(state, ctx)
        if ctx['i'] == 0:
            return ctx['start_idx'], state
        p_idx, e_idx = state['p_idx'], state['e_idx']
        util_ewma = _ewma_step(_norm_proxy(ctx), state.get('util_ewma', 1.0), alpha)
        target_ratio = min(util_ewma * headroom, 1.0)
        perf_limit = ctx['prev_t'][p_idx[-1]] * (1.0 + perf_slack)
        e_feasible = sorted([c for c in e_idx if ctx['prev_t'][c] <= perf_limit])
        action = _freq_select(e_feasible, target_ratio) if e_feasible else _freq_select(p_idx, target_ratio)
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
        metric_independent=True,
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
        state = _ensure_pe_indices(state, ctx)
        if ctx['i'] == 0:
            return ctx['start_idx'], state
        p_idx, e_idx = state['p_idx'], state['e_idx']
        compute_score = np.clip((np.mean(ctx['proxy_window']) - 1.0) / 2.5, 0.0, 1.0)
        util_ewma = _ewma_step(_norm_proxy(ctx), state.get('util_ewma', 1.0), alpha)
        target_ratio = min(util_ewma * headroom, 1.0)
        action = _freq_select(p_idx if compute_score >= compute_thresh else e_idx, target_ratio)
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
        metric_independent=True,
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


# ==========================================
# ISO-FREQ MODEL (Reactive, Model, P↔E cross-proc CatBoost)
# Decides between P-core and E-core at a fixed frequency using cross-processor
# CatBoost models trained on P↔E migrations.  Only the prior chunk's PMU data
# (captured at the current source config) drives the decision.
# ==========================================
def make_isofreq_model(target_freq):
    """Cross-proc model policy: picks P or E at target_freq using CatBoost.

    Returned policy signature (note extra cross_proc_time_mat arg):
        policy(time_mat, energy_mat, proxy_signal, configs, valid_configs,
               trans_lat, trans_nrg, cross_proc_time_mat, metric) -> trace

    cross_proc_time_mat: ndarray (8, n_chunks, n_configs) from
        data_loader._load_cross_proc_time_mat; axis 0 follows ALL_MODEL_CONFIGS.
    """
    def idx_list_fn(configs, valid_configs):
        return sorted([
            configs.index(c) for c in valid_configs
            if c.endswith(target_freq) and c in ALL_MODEL_CONFIGS
        ])

    def start_idx_fn(idx_list, valid_configs):
        # In our configs list E-cores precede P-cores, so sorted idx_list puts
        # P-core last.  Fall back to the last index if only one config exists.
        return len(idx_list) - 1

    return make_model_policy_from_idx_list(
        idx_list_fn=idx_list_fn,
        decision_mode='greedy',
        temporal='reactive',
        src_idx_fn=_src_cfg_idx,
        start_idx_fn=start_idx_fn,
    )


# ==========================================
# REACTIVE ORACLE (fixed freq): prior chunk TRUE timing to pick P vs E
# ==========================================
def make_reactive_oracle_fixed_freq(target_freq):
    """Reactive oracle at fixed frequency: prior chunk's TRUE timing → P or E."""
    return make_policy_from_idx_list(
        idx_list_fn=_fixed_freq_idx_list_fn(target_freq),
        temporal_mode='reactive_oracle',
        decision_mode='greedy',
        window_size=1,
        start_idx_fn=_fixed_freq_start_idx_fn(target_freq),
    )


# ==========================================
# ISOFREQ MODEL ORACLE: current chunk PMU → P vs E (perfect-future model)
# ==========================================
def make_isofreq_model_oracle(target_freq):
    """Cross-proc model with oracle temporal: current chunk PMU → P or E."""
    def idx_list_fn(configs, valid_configs):
        return sorted([
            configs.index(c) for c in valid_configs
            if c.endswith(target_freq) and c in ALL_MODEL_CONFIGS
        ])

    def start_idx_fn(idx_list, valid_configs):
        return len(idx_list) - 1

    return make_model_policy_from_idx_list(
        idx_list_fn=idx_list_fn,
        decision_mode='greedy',
        temporal='oracle',
        src_idx_fn=_src_cfg_idx,
        start_idx_fn=start_idx_fn,
    )


def make_isofreq_model_oracle_k(target_freq, k=1):
    """Cross-proc model oracle with k-step lookahead: averages chunks i..i+k."""
    def idx_list_fn(configs, valid_configs):
        return sorted([
            configs.index(c) for c in valid_configs
            if c.endswith(target_freq) and c in ALL_MODEL_CONFIGS
        ])

    def start_idx_fn(idx_list, valid_configs):
        return len(idx_list) - 1

    return make_model_policy_from_idx_list(
        idx_list_fn=idx_list_fn,
        decision_mode='greedy',
        temporal='oracle',
        src_idx_fn=_src_cfg_idx,
        start_idx_fn=start_idx_fn,
        lookahead_k=k,
    )


# ==========================================
# ISOFREQ ORACLE HEURISTIC: EAS-style with current chunk's TRUE timing
# ==========================================
def make_isofreq_oracle_heuristic(target_freq, perf_slack=0.10):
    """Oracle-signal EAS at fixed freq: uses current chunk's true timing+energy."""
    def decide(ctx, state):
        if ctx['i'] == 0:
            return ctx['start_idx'], state
        # Fixed-freq subset: local 0=E, 1=P (sorted by global config idx, E<P)
        e_local, p_local = 0, 1
        p_time = ctx['prev_t'][p_local]
        e_time = ctx['prev_t'][e_local]
        if e_time <= p_time * (1.0 + perf_slack):
            action = e_local if ctx['prev_e'][e_local] <= ctx['prev_e'][p_local] else p_local
        else:
            action = p_local
        return action, state

    def batch(proxy, sub_t, sub_e, sub_lat, sub_nrg, n, start_idx, vc, il):
        # oracle_heuristic: prev_t/prev_e = sub_t[i, :] / sub_e[i, :]
        e_l, p_l = 0, 1  # E-core is local index 0, P-core is 1 (configs sorted E < P)
        e_ok = sub_t[:, e_l] <= sub_t[:, p_l] * (1.0 + perf_slack)
        e_cheap = sub_e[:, e_l] <= sub_e[:, p_l]
        acts = np.where(e_ok & e_cheap, e_l, p_l).astype(int)
        acts[0] = start_idx
        return acts

    return make_policy_from_idx_list(
        idx_list_fn=_fixed_freq_idx_list_fn(target_freq),
        temporal_mode='oracle_heuristic',
        decision_mode='heuristic',
        window_size=1,
        start_idx_fn=_fixed_freq_start_idx_fn(target_freq),
        heuristic_fn=decide,
        metric_independent=True,
        batch_decide_fn=batch,
    )


# ==========================================
# GREEDY ORACLE HETERO: perfect-future greedy across full P+E space
# ==========================================
def make_greedy_oracle_hetero():
    """Perfect-future 1-step greedy across the full P+E x freq grid."""
    return make_policy_from_idx_list(
        idx_list_fn=_full_idx_list_fn,
        temporal_mode='oracle',
        decision_mode='greedy',
        window_size=1,
        start_idx_fn=_p_max_start_idx,
    )


# ==========================================
# HETERO MODEL REACTIVE/ORACLE: full P+E space via CatBoost full_model_time_mat
# full_model_time_mat: (8, n_chunks, n_configs) combining cross-freq P->P +
# cross-proc P<->E predictions. Passed as model_time_mat positional arg.
# ==========================================
def make_hetero_model_reactive():
    """Full P+E model reactive: prior chunk PMU → best config via CatBoost."""
    return make_model_policy_from_idx_list(
        idx_list_fn=_full_idx_list_fn,
        decision_mode='greedy',
        temporal='reactive',
        src_idx_fn=_src_cfg_idx,
        start_idx_fn=_p_max_start_idx,
    )


def make_hetero_model_oracle():
    """Full P+E model oracle: current chunk PMU → best config via CatBoost."""
    return make_model_policy_from_idx_list(
        idx_list_fn=_full_idx_list_fn,
        decision_mode='greedy',
        temporal='oracle',
        src_idx_fn=_src_cfg_idx,
        start_idx_fn=_p_max_start_idx,
    )


def make_hetero_model_oracle_k(k=1):
    """Full P+E model oracle with k-step lookahead: averages chunks i..i+k."""
    return make_model_policy_from_idx_list(
        idx_list_fn=_full_idx_list_fn,
        decision_mode='greedy',
        temporal='oracle',
        src_idx_fn=_src_cfg_idx,
        start_idx_fn=_p_max_start_idx,
        lookahead_k=k,
    )


# ==========================================
# ORACLE HEURISTICS (Hetero): EAS / Thread Director with current chunk's TRUE data
# ==========================================
def make_eas_oracle(alpha=0.25, perf_slack=0.10):
    """Oracle-signal EAS: full P+E space, current chunk's true timing+energy."""
    def decide(ctx, state):
        state = _ensure_pe_indices(state, ctx)
        if ctx['i'] == 0:
            return ctx['start_idx'], state
        p_idx = state['p_idx']
        feasible = _perf_feasible(ctx, perf_slack, p_idx[-1]) or p_idx
        action = min(feasible, key=lambda c: ctx['prev_e'][c])
        state = dict(state)
        return action, state

    return make_policy_from_idx_list(
        idx_list_fn=_full_idx_list_fn,
        temporal_mode='oracle_heuristic',
        decision_mode='heuristic',
        window_size=1,
        start_idx_fn=_p_max_start_idx,
        heuristic_fn=decide,
        metric_independent=True,
    )


def make_thread_director_oracle(compute_thresh=0.60, headroom=1.25):
    """Oracle-signal Thread Director: current chunk's proxy + timing."""
    def decide(ctx, state):
        state = _ensure_pe_indices(state, ctx)
        if ctx['i'] == 0:
            return ctx['start_idx'], state
        p_idx, e_idx = state['p_idx'], state['e_idx']
        compute_score = _norm_proxy(ctx)
        target_ratio = min(compute_score * headroom, 1.0)
        action = _freq_select(p_idx if compute_score >= compute_thresh else e_idx, target_ratio)
        state = dict(state)
        return action, state

    def batch(proxy, sub_t, sub_e, sub_lat, sub_nrg, n, start_idx, vc, il):
        # oracle_heuristic: proxy_window[-1] = proxy_signal[i] for each chunk i
        p_idx = np.array([k for k, c in enumerate(vc) if c.startswith('P')])
        e_idx = np.array([k for k, c in enumerate(vc) if c.startswith('E')])
        score = np.clip((proxy - 1.0) / 2.5, 0.0, 1.0)
        ratio = np.minimum(score * headroom, 1.0)
        n_p, n_e = len(p_idx), len(e_idx)
        p_lvl = np.clip(np.round(ratio * (n_p - 1)), 0, n_p - 1).astype(int)
        e_lvl = np.clip(np.round(ratio * (n_e - 1)), 0, n_e - 1).astype(int)
        acts = np.where(score >= compute_thresh, p_idx[p_lvl], e_idx[e_lvl]).astype(int)
        acts[0] = start_idx
        return acts

    return make_policy_from_idx_list(
        idx_list_fn=_full_idx_list_fn,
        temporal_mode='oracle_heuristic',
        decision_mode='heuristic',
        window_size=1,
        start_idx_fn=_p_max_start_idx,
        heuristic_fn=decide,
        metric_independent=True,
        batch_decide_fn=batch,
    )

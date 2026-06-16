# dvfs_policies.py
#
# Single-core (P or E) DVFS policies. Each is now a thin wrapper around
# decision_policies.make_policy_from_idx_list - the temporal axis (Reactive
# vs Oracle) and decision axis (Greedy/MPC/Global/Heuristic) are supplied by
# decision_policies.py / temporal_policies.py.
import numpy as np

from decision_policies import make_policy_from_idx_list, make_model_policy_from_idx_list


def _core_idx_list_fn(core_type):
    def idx_list_fn(configs, valid_configs):
        return sorted([configs.index(c) for c in valid_configs if c.startswith(core_type)])
    return idx_list_fn


# ==========================================
# 1. STATIC PINNED POLICIES
# ==========================================
def _decide_heuristic_static(ctx, state):
    return 0, state


def make_static(target_cfg):
    def idx_list_fn(configs, valid_configs):
        return [configs.index(target_cfg)]

    return make_policy_from_idx_list(
        idx_list_fn=idx_list_fn,
        temporal_mode='reactive',
        decision_mode='heuristic',
        window_size=1,
        start_idx_fn=lambda idx_list, valid_configs: 0,
        heuristic_fn=_decide_heuristic_static,
    )


# ==========================================
# 2. REACTIVE 1-TIMESTEP (Greedy, Reactive)
# ==========================================
def make_reactive_1_step(core_type):
    return make_policy_from_idx_list(
        idx_list_fn=_core_idx_list_fn(core_type),
        temporal_mode='reactive',
        decision_mode='greedy',
        window_size=1,
    )


# ==========================================
# 3. PROACTIVE 1-TIMESTEP (Greedy, Oracle)
# ==========================================
def make_proactive_1_step(core_type):
    return make_policy_from_idx_list(
        idx_list_fn=_core_idx_list_fn(core_type),
        temporal_mode='oracle',
        decision_mode='greedy',
        window_size=1,
    )


# ==========================================
# 4. REACTIVE ORACLE (Greedy, Reactive-Oracle)
# Uses prior chunk's TRUE timings to decide current chunk's frequency.
# Perfect knowledge of the past; no knowledge of the future.
# ==========================================
def make_reactive_oracle(core_type):
    return make_policy_from_idx_list(
        idx_list_fn=_core_idx_list_fn(core_type),
        temporal_mode='reactive_oracle',
        decision_mode='greedy',
        window_size=1,
    )


# ==========================================
# 4. GLOBAL PROACTIVE ORACLE (Global, Oracle)
# ==========================================
def make_global_viterbi(core_type):
    return make_policy_from_idx_list(
        idx_list_fn=_core_idx_list_fn(core_type),
        temporal_mode='oracle',
        decision_mode='global',
    )


# ==========================================
# PROACTIVE N-STEP MPC (MPC, Oracle)
# ==========================================
def make_proactive_n_step(core_type, horizon):
    return make_policy_from_idx_list(
        idx_list_fn=_core_idx_list_fn(core_type),
        temporal_mode='oracle',
        decision_mode='mpc',
        window_size=horizon,
    )


# ==========================================
# 5. LINUX PERFORMANCE GOVERNOR (Heuristic, Reactive)
# ==========================================
def _decide_heuristic_performance_governor(ctx, state):
    n = ctx['sub_t'].shape[1]
    return n - 1, state


def run_performance_governor(time_mat, energy_mat, proxy_signal, configs, valid_configs, trans_lat, trans_nrg, metric):
    """Linux 'performance' CPUfreq governor: always pins to maximum frequency."""
    policy = make_policy_from_idx_list(
        idx_list_fn=_core_idx_list_fn('P'),
        temporal_mode='reactive',
        decision_mode='heuristic',
        window_size=1,
        heuristic_fn=_decide_heuristic_performance_governor,
    )
    return policy(time_mat, energy_mat, proxy_signal, configs, valid_configs, trans_lat, trans_nrg, metric)


# ==========================================
# 6. INTEL HWP (Heuristic, Reactive)
# ==========================================
def _decide_heuristic_intel_hwp(ctx, state):
    if ctx['i'] == 0:
        return ctx['start_idx'], state
    n = ctx['sub_t'].shape[1]
    raw_proxy = ctx['proxy_window'][-1]
    hwp_ratio = np.clip((raw_proxy - 1.0) / 2.5, 0.0, 1.0)
    hwp_target = int(hwp_ratio * (n - 1))
    return hwp_target, state


def run_intel_hwp(time_mat, energy_mat, proxy_signal, configs, valid_configs, trans_lat, trans_nrg, metric):
    policy = make_policy_from_idx_list(
        idx_list_fn=_core_idx_list_fn('P'),
        temporal_mode='reactive',
        decision_mode='heuristic',
        window_size=1,
        heuristic_fn=_decide_heuristic_intel_hwp,
    )
    return policy(time_mat, energy_mat, proxy_signal, configs, valid_configs, trans_lat, trans_nrg, metric)


# ==========================================
# 7. LINUX ONDEMAND GOVERNOR (Heuristic, Reactive)
# ==========================================
def make_ondemand(core_type, up_thresh=0.80, down_thresh=0.20):
    """
    Linux 'ondemand' CPUfreq governor (Pallipadi & Starikovskiy, OLS 2006).
    Jumps to max frequency when normalized proxy utilization exceeds up_thresh;
    drops to min frequency when below down_thresh; otherwise holds current level.
    Proxy signal is normalized to [0,1] via clip((proxy-1)/2.5).
    """
    def decide(ctx, state):
        if ctx['i'] == 0:
            return ctx['start_idx'], state
        n = ctx['sub_t'].shape[1]
        util = np.clip((ctx['proxy_window'][-1] - 1.0) / 2.5, 0.0, 1.0)
        if util >= up_thresh:
            return n - 1, state
        elif util <= down_thresh:
            return 0, state
        return ctx['prev_idx'], state

    return make_policy_from_idx_list(
        idx_list_fn=_core_idx_list_fn(core_type),
        temporal_mode='reactive',
        decision_mode='heuristic',
        window_size=1,
        heuristic_fn=decide,
    )


# ==========================================
# 8. LINUX CONSERVATIVE GOVERNOR (Heuristic, Reactive)
# ==========================================
def make_conservative(core_type, up_thresh=0.80, down_thresh=0.20):
    """
    Linux 'conservative' CPUfreq governor.
    Steps frequency up or down by exactly one level per interval, avoiding the
    large transition costs of ondemand's jumps to max/min.
    Reference: Linux kernel CPUfreq documentation.
    """
    def decide(ctx, state):
        n = ctx['sub_t'].shape[1]
        level = state.get('level', n - 1)
        if ctx['i'] == 0:
            return level, state
        util = np.clip((ctx['proxy_window'][-1] - 1.0) / 2.5, 0.0, 1.0)
        if util >= up_thresh:
            level = min(level + 1, n - 1)
        elif util <= down_thresh:
            level = max(level - 1, 0)
        return level, {'level': level}

    return make_policy_from_idx_list(
        idx_list_fn=_core_idx_list_fn(core_type),
        temporal_mode='reactive',
        decision_mode='heuristic',
        window_size=1,
        heuristic_fn=decide,
    )


# ==========================================
# 9. LINUX SCHEDUTIL (PELT-Based) (Heuristic, Reactive)
# ==========================================
def make_schedutil_pelt(core_type, alpha=0.25, headroom=1.25):
    """
    Linux 'schedutil' governor with PELT-style EWMA utilization tracking
    (Linux kernel v4.7+; Rafał Miłecki, Viresh Kumar, Rafael Wysocki).
    Smooths the proxy utilization signal via EWMA, then selects the minimum
    frequency level satisfying: freq >= util_ewma * headroom * max_freq.
    alpha controls EWMA decay; headroom (>1.0) reserves margin above predicted load.
    """
    def decide(ctx, state):
        if ctx['i'] == 0:
            return ctx['start_idx'], state
        n = ctx['sub_t'].shape[1]
        util_ewma = state.get('util_ewma', 1.0)
        raw_util = np.clip((ctx['proxy_window'][-1] - 1.0) / 2.5, 0.0, 1.0)
        util_ewma = alpha * raw_util + (1.0 - alpha) * util_ewma
        target_ratio = min(util_ewma * headroom, 1.0)
        target_level = int(np.round(target_ratio * (n - 1)))
        return int(np.clip(target_level, 0, n - 1)), {'util_ewma': util_ewma}

    return make_policy_from_idx_list(
        idx_list_fn=_core_idx_list_fn(core_type),
        temporal_mode='reactive',
        decision_mode='heuristic',
        window_size=1,
        heuristic_fn=decide,
    )


# ==========================================
# 10. EWMA PREDICTOR (Weiser et al., 1994) (Heuristic, Reactive)
# ==========================================
def make_ewma_dvfs(core_type, alpha=0.5):
    """
    EWMA-based DVFS from Weiser et al., "Scheduling for Reduced CPU Energy" (OSDI 1994).
    Predicts next-interval utilization via exponentially weighted moving average of the
    proxy signal, then runs at the minimum frequency sufficient for that prediction.
    alpha controls how quickly the predictor tracks changes (higher = more reactive).
    """
    def decide(ctx, state):
        if ctx['i'] == 0:
            return ctx['start_idx'], state
        n = ctx['sub_t'].shape[1]
        pred = state.get('pred', 1.0)
        actual = np.clip((ctx['proxy_window'][-1] - 1.0) / 2.5, 0.0, 1.0)
        pred = alpha * actual + (1.0 - alpha) * pred
        target_level = int(np.ceil(pred * (n - 1)))
        return int(np.clip(target_level, 0, n - 1)), {'pred': pred}

    return make_policy_from_idx_list(
        idx_list_fn=_core_idx_list_fn(core_type),
        temporal_mode='reactive',
        decision_mode='heuristic',
        window_size=1,
        heuristic_fn=decide,
    )


# ==========================================
# 11. UCB1 ONLINE BANDIT DVFS (Heuristic, Reactive)
# ==========================================
def make_ucb1_dvfs(core_type, c=1.0):
    """
    UCB1 multi-armed bandit DVFS (Auer et al., Machine Learning 2002).
    Treats each available frequency as a bandit arm; reward is the negative
    per-chunk EDP/ED2P cost, normalized online by the running maximum to keep
    UCB confidence bounds scaled correctly across workloads.
    Initialization: round-robins through all arms once before exploiting.
    c controls the exploration-exploitation tradeoff (larger = more exploration).
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
        idx_list_fn=_core_idx_list_fn(core_type),
        temporal_mode='reactive',
        decision_mode='heuristic',
        window_size=1,
        heuristic_fn=decide,
    )


# ==========================================
# 12. RANDOM POLICY (Lower Bound) (Heuristic, Reactive)
# ==========================================
def make_random_dvfs(core_type, seed=42):
    """
    Random frequency selection within the specified core cluster.
    Serves as a lower bound for online learning policies (UCB1, EWMA).
    A fixed seed ensures reproducibility across workloads and metric types.
    """
    def decide(ctx, state):
        n = ctx['sub_t'].shape[1]
        rng = state['rng']
        return int(rng.integers(n)), state

    return make_policy_from_idx_list(
        idx_list_fn=_core_idx_list_fn(core_type),
        temporal_mode='reactive',
        decision_mode='heuristic',
        window_size=1,
        heuristic_fn=decide,
        initial_state=lambda: {'rng': np.random.default_rng(seed)},
    )


# ==========================================
# Model-based policies (decision axis: Model)
# Use cross-freq CatBoost predictions to decide between P-core frequencies.
# Returned policy takes model_time_mat as an extra positional argument before metric.
# ==========================================

def _p_model_idx_list_fn():
    """Restrict action set to P_1.0-4.0GHz (exclude P_5.0GHz and all E-cores)."""
    def idx_list_fn(configs, valid_configs):
        from data_loader import P_MODEL_FREQS
        allowed = {f"P_{f:.1f}GHz" for f in P_MODEL_FREQS}
        return sorted([configs.index(c) for c in valid_configs if c in allowed])
    return idx_list_fn


def make_model_1_step(core_type='P'):
    """Model-based Greedy: uses CatBoost cross-freq predictions to pick next freq."""
    return make_model_policy_from_idx_list(
        idx_list_fn=_p_model_idx_list_fn(),
        decision_mode='greedy',
        window_size=1,
    )


def make_model_n_step(core_type='P', horizon=5):
    """Model-based MPC: predicts over a W-chunk horizon then picks first action."""
    return make_model_policy_from_idx_list(
        idx_list_fn=_p_model_idx_list_fn(),
        decision_mode='mpc',
        window_size=horizon,
    )


def make_model_global(core_type='P'):
    """Model-based Global Viterbi: full-trace DP using cross-freq model predictions."""
    return make_model_policy_from_idx_list(
        idx_list_fn=_p_model_idx_list_fn(),
        decision_mode='global',
    )

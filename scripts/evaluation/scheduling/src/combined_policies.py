# combined_policies.py
#
# Combined core-placement + per-core DVFS (full P+E x freq grid) policies.
# Thin wrappers around decision_policies.make_policy_from_idx_list.
from decision_policies import make_policy_from_idx_list
from scheduling_policies import _full_idx_list_fn, _p_max_start_idx


# ==========================================
# 1. REACTIVE N-STEP COMBINED (MPC, Reactive, History Lookback)
# ==========================================
def make_reactive_n_step_combined(lookback):
    return make_policy_from_idx_list(
        idx_list_fn=_full_idx_list_fn,
        temporal_mode='reactive',
        decision_mode='mpc',
        window_size=lookback,
        start_idx_fn=_p_max_start_idx,
    )


# ==========================================
# 2. PROACTIVE N-STEP COMBINED (MPC, Oracle, MPC Lookahead)
# ==========================================
def make_proactive_n_step_combined(horizon):
    return make_policy_from_idx_list(
        idx_list_fn=_full_idx_list_fn,
        temporal_mode='oracle',
        decision_mode='mpc',
        window_size=horizon,
        start_idx_fn=_p_max_start_idx,
    )

import argparse
import hashlib
import json
import os
import re
import numpy as np
from pathlib import Path
import concurrent.futures

# Direct local imports (since they are in the same folder as main.py)
import data_loader
from data_loader import (load_phase_data, load_workload_data, _load_model_time_mat,
                         _load_e_model_time_mat, _load_cross_proc_time_mat,
                         _load_full_model_time_mat, build_oracle_axis_mat,
                         _load_forecast_oracle_mat)
from decision_policies import compute_trace_stats, accumulate_trace
import dvfs_policies as dvfs
import scheduling_policies as sched
import plotter
import combined_policies as comb
import warmup_model


# v2: all policies (incl. the Viterbi oracles) now start pinned to the same config.
VITERBI_CACHE_VERSION = 2


def _call_with_stats(policy_fn, *args, **kwargs):
    """Call a policy, returning (trace, stats_dict | None)."""
    if getattr(policy_fn, 'returns_actions', False):
        trace, actions, local_names = policy_fn(*args, _return_actions=True, **kwargs)
        return trace, compute_trace_stats(list(actions), local_names)
    return policy_fn(*args, **kwargs), None


def _record_diag(diag_results, wl, ph, m_type, name, st):
    if st is None:
        return
    diag_results.append({
        'Workload': wl, 'Phase': ph, 'Metric': m_type, 'Policy': name,
        **{k: v for k, v in st.items() if not k.startswith('frac_')},
        'config_fracs': json.dumps({k[5:]: round(v, 4)
                                     for k, v in st.items() if k.startswith('frac_')}),
    })

# ==========================================
# GLOBAL HARDWARE DEFINES
# ==========================================
# Cache warm-up penalty parameters (fill in from warmup_params.csv after measurement).
# Set both A values to 0.0 to verify results are identical to the no-penalty baseline.
# Estimates for 10M-instruction chunks on Intel Alder Lake:
#   L3 is shared between P/E clusters, so only L1d/L2 + branch predictor + hw prefetchers
#   go cold after migration.  Dominant costs at this chunk size:
#     L2 cache cold-start (1-4M instruction warmup):   ~10-15% peak slowdown
#     BTB/BHT miss spike (200K-1M instruction warmup): ~3-8% peak slowdown (gcc/perlbench worst)
#     HW prefetcher retraining (50-200K accesses):     ~3-5% peak slowdown (memory-bound)
#   P→E is larger: E-core shares 2MB L2 across 4 cores (~0.5MB effective) vs 1.25MB on P,
#   and Golden Cove has more aggressive prefetch/BTB units that warm faster on E→P.
#   tau in CHUNK units (1 chunk = 10M instructions ≈ 1-3ms at 3-4GHz):
#     warmup completes in ~3-4M instructions = 0.3-0.4 chunks → tau ≈ 0.4 P→E, 0.3 E→P.
#   Replace with warmup_collection.sh measurements when available.
WARMUP_A_PtoE   = 0.20  # amplitude:  P→E peak slowdown (~20% at migration point)
WARMUP_TAU_PtoE = 0.4   # decay time: P→E (chunks; warmup ~3-4M instr = 0.3-0.4 chunks)
WARMUP_A_EtoP   = 0.12  # amplitude:  E→P peak slowdown (~12%; P-core warms faster)
WARMUP_TAU_EtoP = 0.3   # decay time: E→P (chunks)
WARMUP_K        = 10    # number of post-migration chunks to penalize (3 tau is enough)

# P↔E context switch: mean 4.47μs, symmetric within 0.1μs (ctx_switch_bench, 10 reps × 5000 migrations)
MIG_LAT_S  = 4.47e-6
# Migration energy (J), keyed by (P_freq, E_freq) of the two cores involved
# in the migration: MIG_NRG_J = migration_active_power_W × MIG_LAT_S, where
# migration_active_power_W is the median measured package power (RAPL
# power/energy-cores) while ctx_switch_bench runs continuous P↔E migrations
# at that frequency pair (power_collection/ctx_switch/freq_sweep_power/,
# analyze_migration_energy.py -> migration_energy_summary.csv).
#
# This replaces an earlier idle-subtraction approach (duty-cycled bench +
# same-core control, see analyze_duty_test.py) that tried to isolate the
# ~1-10nJ energy of a single migration as a delta against RAPL's ~50ms /
# ~10-50mW noise floor and produced sign-flipping, order-of-magnitude-unstable
# results. Using the directly-measured *active* power (O(1-25W), the same
# convention as P_NRG_J/E_NRG_J's Trans_P95_W × stall_latency) avoids that
# noise floor entirely, at the cost of including some non-migration-specific
# background power in the estimate.
MIG_NRG_J = {
    (1.0, 1.0): 5.35e-06,  (1.0, 2.0): 8.92e-06,  (1.0, 3.0): 2.139e-05, (1.0, 4.0): 8.909e-05,
    (2.0, 1.0): 8.93e-06,  (2.0, 2.0): 1.25e-05,  (2.0, 3.0): 2.589e-05, (2.0, 4.0): 9.637e-05,
    (3.0, 1.0): 1.875e-05, (3.0, 2.0): 2.054e-05, (3.0, 3.0): 2.852e-05, (3.0, 4.0): 0.00010253,
    (4.0, 1.0): 4.012e-05, (4.0, 2.0): 5.893e-05, (4.0, 3.0): 5.974e-05, (4.0, 4.0): 0.0001061,
}
DVFS_LAT_S = 5.0e-6    # fallback for freq pairs not in P_LAT_US/E_LAT_US
DVFS_NRG_J = 2e-5      # fallback


# DVFS transition latencies (μs): simple_latency.c + analyze_dvfs.py, 100 reps each.
# Median across reps (robust to the long right tail seen on some downscale/E-core
# transitions, where the core occasionally steps through an intermediate P-state).
# Entries that measured 0.00 (transition completes within a single ~1 sample
# compute kernel, below the rolling-median detection resolution) are floored to
# one sample period at the target frequency: P-core 3GHz=2.68us, E-core 4GHz=3.01us,
# E-core 3GHz=4.02us.
P_LAT_US = {
    (1.0, 2.0): 4.02,  (1.0, 3.0): 2.68,  (1.0, 4.0): 2.01,
    (2.0, 1.0): 7.93,  (2.0, 3.0): 2.68,  (2.0, 4.0): 2.01,
    (3.0, 1.0): 8.04,  (3.0, 2.0): 2.68,  (3.0, 4.0): 2.01,
    (4.0, 1.0): 8.04,  (4.0, 2.0): 2.22,  (4.0, 3.0): 2.68,  # floored (median 0.00)
}
E_LAT_US = {
    (1.0, 2.0): 6.03,  (1.0, 3.0): 8.02,  (1.0, 4.0): 6.04,
    (2.0, 1.0): 12.04, (2.0, 3.0): 4.02,  (2.0, 4.0): 3.01,
    (3.0, 1.0): 12.04, (3.0, 2.0): 4.01,  (3.0, 4.0): 3.01,  # floored (median 0.00)
    (4.0, 1.0): 12.06, (4.0, 2.0): 3.69,  (4.0, 3.0): 4.02,  # floored (median 0.00)
}

# DVFS transition energy (J) = Trans_P95_W × stall_latency
P_NRG_J = {
    (1.0, 2.0): 2.1e-05,  (1.0, 3.0): 1.6e-05,  (1.0, 4.0): 1.3e-05,
    (2.0, 1.0): 5.8e-05,  (2.0, 3.0): 2.1e-05,  (2.0, 4.0): 1.8e-05,
    (3.0, 1.0): 8.4e-05,  (3.0, 2.0): 5.7e-05,  (3.0, 4.0): 2.6e-05,
    (4.0, 1.0): 0.000117, (4.0, 2.0): 6.6e-05,  (4.0, 3.0): 4.7e-05,
}
E_NRG_J = {
    (1.0, 2.0): 2.3e-05,  (1.0, 3.0): 1.6e-05,  (1.0, 4.0): 1.5e-05,
    (2.0, 1.0): 5.8e-05,  (2.0, 3.0): 2.2e-05,  (2.0, 4.0): 2.0e-05,
    (3.0, 1.0): 7.1e-05,  (3.0, 2.0): 4.4e-05,  (3.0, 4.0): 2.9e-05,
    (4.0, 1.0): 0.0002,   (4.0, 2.0): 0.000132, (4.0, 3.0): 9.7e-05,
}

# --- ARM big.LITTLE (Qualcomm RB5 / Snapdragon 865) placeholder parameters ---
# L = Little (Silver, Cortex-A55), B = Big (Gold, Cortex-A76)
# These are estimates; replace with measured values when available.
ARM_WARMUP_A_BtoL   = 0.20
ARM_WARMUP_TAU_BtoL = 0.4
ARM_WARMUP_A_LtoB   = 0.12
ARM_WARMUP_TAU_LtoB = 0.3
ARM_WARMUP_K        = 10

ARM_MIG_LAT_S  = 5.0e-6
ARM_MIG_NRG_J  = {
    (1.0, 1.0): 5.0e-06,
}
ARM_DVFS_LAT_S = 5.0e-6
ARM_DVFS_NRG_J = 2e-5

ARM_L_LAT_US = {}
ARM_L_NRG_J  = {}
ARM_B_LAT_US = {}
ARM_B_NRG_J  = {}


def get_dvfs_cost(start_cfg, end_cfg):
    """Calculates true Latency (s) and Energy (J) for a specific frequency hop."""
    s_type, s_freq_str = start_cfg.split('_')
    e_type, e_freq_str = end_cfg.split('_')
    
    s_freq = float(s_freq_str.replace('GHz', ''))
    e_freq = float(e_freq_str.replace('GHz', ''))
    
    if s_freq == e_freq:
        return 0.0, 0.0
        
    lat_map = {'P': P_LAT_US, 'E': E_LAT_US, 'B': ARM_B_LAT_US, 'L': ARM_L_LAT_US}[s_type]
    nrg_map = {'P': P_NRG_J, 'E': E_NRG_J, 'B': ARM_B_NRG_J, 'L': ARM_L_NRG_J}[s_type]
    
    # Fallback to defaults if a missing transition is requested
    lat_us = lat_map.get((s_freq, e_freq), DVFS_LAT_S * 1e6)
    nrg_j = nrg_map.get((s_freq, e_freq), DVFS_NRG_J)
    
    lat_s = lat_us * 1e-6

    return lat_s, nrg_j

def get_migration_cost(start_cfg, end_cfg):
    """Calculates Latency (s) and Energy (J) for a cross-cluster migration."""
    s_type, s_freq_str = start_cfg.split('_')
    e_type, e_freq_str = end_cfg.split('_')

    if s_type in ('L', 'B'):
        b_freq_str = s_freq_str if s_type == 'B' else e_freq_str
        l_freq_str = e_freq_str if s_type == 'B' else s_freq_str
        b_freq = float(b_freq_str.replace('GHz', ''))
        l_freq = float(l_freq_str.replace('GHz', ''))
        nrg_j = ARM_MIG_NRG_J.get((b_freq, l_freq), ARM_DVFS_NRG_J)
        return ARM_MIG_LAT_S, nrg_j

    p_freq_str = s_freq_str if s_type == 'P' else e_freq_str
    e_freq_str_val = e_freq_str if s_type == 'P' else s_freq_str

    p_freq = min(float(p_freq_str.replace('GHz', '')), 4.0)
    e_freq = min(float(e_freq_str_val.replace('GHz', '')), 4.0)

    nrg_j = MIG_NRG_J.get((p_freq, e_freq), DVFS_NRG_J)
    return MIG_LAT_S, nrg_j

def build_transition_matrices(configs):
    n = len(configs)
    lat_mat, nrg_mat = np.zeros((n, n)), np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j: continue
            
            # Cross-cluster Migration Cost
            if configs[i][0] != configs[j][0]:
                lat_mat[i, j], nrg_mat[i, j] = get_migration_cost(configs[i], configs[j])
            # Same-cluster DVFS Cost
            else:
                lat_s, nrg_j = get_dvfs_cost(configs[i], configs[j])
                lat_mat[i, j], nrg_mat[i, j] = lat_s, nrg_j
                    
    return lat_mat, nrg_mat

def process_workload(wl, ph, pairs, input_path, configs,
                     power_mode='per_sample', cross_freq_p_pred_dir=None,
                     cross_freq_e_pred_dir=None, cross_proc_pred_dir=None,
                     cross_freq_p_forecast_dir=None, cross_freq_e_forecast_dir=None,
                     cross_proc_forecast_dir=None, forecast_oracle_dir=None,
                     cross_freq_p_forecast_gated_dir=None,
                     cross_freq_e_forecast_gated_dir=None,
                     cross_proc_forecast_gated_dir=None,
                     forecast_oracle_gated_dir=None,
                     cross_freq_p_forecast_unaware_dir=None,
                     cross_freq_e_forecast_unaware_dir=None,
                     cross_proc_forecast_unaware_dir=None,
                     forecast_oracle_unaware_dir=None,
                     viterbi_cache_dir=None, apply_warmup=False, phases=None,
                     pred_tag='none', env_tag='none'):
    if phases is not None:
        data = load_workload_data(wl, phases, input_path, configs, power_mode=power_mode,
                                  model_pred_dir=cross_freq_p_pred_dir,
                                  e_model_pred_dir=cross_freq_e_pred_dir,
                                  cross_proc_pred_dir=cross_proc_pred_dir)
    else:
        data = load_phase_data(wl, ph, input_path, configs, power_mode=power_mode,
                               model_pred_dir=cross_freq_p_pred_dir,
                               e_model_pred_dir=cross_freq_e_pred_dir,
                               cross_proc_pred_dir=cross_proc_pred_dir)
    if data is None: return None

    time_mat, energy_mat, proxy_signal, valid_configs, min_len, model_time_mat, e_model_time_mat, cross_proc_time_mat, full_model_time_mat = data
    min_len = len(time_mat)

    # Walk-forward FORECAST tensors: same source-keyed layout as the reactive/oracle
    # model predictions, but each row i is a *causal forecast* for chunk i (made from
    # history < i). Built here (per-phase path) via the same loaders; row-i indexing
    # (make_model_1_step_oracle) is therefore an honest deployable forecast, not a peek.
    #
    # Two variants may be supplied. They differ ONLY in how the forecaster was trained
    # (dump_dvfs_forecast.py --method per_phase vs global), so the file layout and every
    # loader below are shared:
    #   'aware'   -- per-phase models overwrite a global model's predictions for chunks
    #                whose runtime phase has enough training samples.
    #   'unaware' -- one model over the trailing window, no phase segmentation.
    # Supplying both lets the scheduler measure whether phase-awareness converts into
    # scheduling gains, rather than assuming the standalone MAPE advantage carries over.
    def _build_forecast_tensors(p_dir, e_dir, cp_dir):
        f_p = f_e = f_cp = f_full = None
        if phases is not None or not (p_dir or e_dir or cp_dir):
            return f_p, f_e, f_cp, f_full
        power_mat = np.divide(energy_mat, time_mat, out=np.zeros_like(energy_mat),
                              where=time_mat != 0)
        if p_dir is not None:
            f_p = _load_model_time_mat(wl, ph, p_dir, configs, min_len, time_mat, power_mat)
        if e_dir is not None:
            f_e = _load_e_model_time_mat(wl, ph, e_dir, configs, min_len, time_mat, power_mat)
        if cp_dir is not None:
            f_cp = _load_cross_proc_time_mat(wl, ph, cp_dir, configs, min_len, time_mat)
        # Merged tensor over all 8 core x freq configs. The E cross-freq forecast must be
        # passed too, or E-core frequency changes are invisible once on the E-core.
        if f_p is not None and f_cp is not None:
            f_full = _load_full_model_time_mat(f_p, f_cp, f_e)
        return f_p, f_e, f_cp, f_full

    (forecast_time_mat, e_forecast_time_mat, cross_proc_forecast_time_mat,
     full_forecast_time_mat) = _build_forecast_tensors(
        cross_freq_p_forecast_dir, cross_freq_e_forecast_dir, cross_proc_forecast_dir)

    (fc_un_p_mat, fc_un_e_mat, _fc_un_cp_mat,
     fc_un_full_mat) = _build_forecast_tensors(
        cross_freq_p_forecast_unaware_dir, cross_freq_e_forecast_unaware_dir,
        cross_proc_forecast_unaware_dir)

    # GATED arm: the forecaster defers to persistence, per chunk, whenever it has not been
    # beating persistence on recent history. Persistence is exactly what the reactive
    # policy reads, so the gate bounds this policy below by reactive -- which is the one
    # thing the ungated forecaster could not do (it lost to reactive on 2 of 3 suites).
    (fc_g_p_mat, fc_g_e_mat, _fc_g_cp_mat,
     fc_g_full_mat) = _build_forecast_tensors(
        cross_freq_p_forecast_gated_dir, cross_freq_e_forecast_gated_dir,
        cross_proc_forecast_gated_dir)

    # Forecast x Oracle: the missing cell of the temporal x prediction-source grid.
    forecast_oracle_mat = None
    if phases is None and forecast_oracle_dir is not None:
        forecast_oracle_mat = _load_forecast_oracle_mat(
            wl, ph, forecast_oracle_dir, configs, min_len, time_mat)
    forecast_oracle_unaware_mat = None
    if phases is None and forecast_oracle_unaware_dir is not None:
        forecast_oracle_unaware_mat = _load_forecast_oracle_mat(
            wl, ph, forecast_oracle_unaware_dir, configs, min_len, time_mat)
    forecast_oracle_gated_mat = None
    if phases is None and forecast_oracle_gated_dir is not None:
        forecast_oracle_gated_mat = _load_forecast_oracle_mat(
            wl, ph, forecast_oracle_gated_dir, configs, min_len, time_mat)

    trans_lat, trans_nrg = build_transition_matrices(configs)

    summary_results = []
    diag_results = []

    # Define common arguments for policy calls
    policy_args = (time_mat, energy_mat, proxy_signal, configs, valid_configs, trans_lat, trans_nrg)

    METRICS = ['EDP', 'ED2P']

    # Policy registries: {name: (fn, args_tuple, metric_kwarg)}
    # Built once; metric_kwarg is passed per-call for metric-dependent policies.

    # 1. P-Core DVFS Policies
    p_calls = {
        # --- Static baselines ---
        'Static_P_1.0GHz':       (dvfs.make_static('P_1.0GHz'),             policy_args),
        'Static_P_2.0GHz':       (dvfs.make_static('P_2.0GHz'),             policy_args),
        'Static_P_3.0GHz':       (dvfs.make_static('P_3.0GHz'),             policy_args),
        'Static_P_4.0GHz':       (dvfs.make_static('P_4.0GHz'),             policy_args),
        # --- Reactive heuristics (decide using prior chunk's proxy) ---
        'Performance_Gov_P':     (dvfs.make_performance_governor('P'),       policy_args),
        'Powersave_Gov_P':       (dvfs.make_powersave_governor('P'),         policy_args),
        'Ondemand_P':            (dvfs.make_ondemand('P'),                   policy_args),
        'Conservative_P':        (dvfs.make_conservative('P'),               policy_args),
        'Schedutil_PELT_P':      (dvfs.make_schedutil_pelt('P'),            policy_args),
        'Interactive_P':         (dvfs.make_interactive('P'),                policy_args),
        'Intel_HWP_P':           (dvfs.make_intel_hwp('P'),                  policy_args),
        'EWMA_P':                (dvfs.make_ewma_dvfs('P'),                  policy_args),
        'UCB1_P':                (dvfs.make_ucb1_dvfs('P'),                  policy_args),
        # --- Reactive oracle (repeats prior chunk's best config) ---
        'Reactive_Oracle_P':     (dvfs.make_reactive_oracle('P'),            policy_args),
        # --- Perfect-future heuristics (same algorithms, current chunk's true proxy) ---
        'Ondemand_Future_P':     (dvfs.make_ondemand('P',     temporal_mode='oracle_heuristic'), policy_args),
        'Conservative_Future_P': (dvfs.make_conservative('P', temporal_mode='oracle_heuristic'), policy_args),
        'Schedutil_Future_P':    (dvfs.make_schedutil_pelt('P', temporal_mode='oracle_heuristic'), policy_args),
        'Interactive_Future_P':  (dvfs.make_interactive('P', temporal_mode='oracle_heuristic'), policy_args),
        'Intel_HWP_Future_P':    (dvfs.make_intel_hwp('P',   temporal_mode='oracle_heuristic'), policy_args),
        'EWMA_Future_P':         (dvfs.make_ewma_dvfs('P',   temporal_mode='oracle_heuristic'), policy_args),
        'UCB1_Future_P':         (dvfs.make_ucb1_dvfs('P',   temporal_mode='oracle_heuristic'), policy_args),
        # --- Oracle bounds ---
        'Greedy_Oracle_P':       (dvfs.make_proactive_1_step('P'),           policy_args),
        'Global_Oracle_P':       (dvfs.make_global_viterbi('P'),             policy_args),
    }
    if model_time_mat is not None:
        model_policy_args = (*policy_args, model_time_mat)
        p_calls.update({
            # --- Reactive model (prior chunk PMU → predicted best config) ---
            'Model_Greedy_P':          (dvfs.make_model_1_step('P'),              model_policy_args),
            # --- Perfect-future model (current chunk PMU, oracle temporal) ---
            'Model_Greedy_Oracle_P':   (dvfs.make_model_1_step_oracle('P'),       model_policy_args),
            # --- Oracle lookahead (average of next k+1 chunks' predictions) ---
            'Model_Greedy_Oracle_k1_P': (dvfs.make_model_1_step_oracle_k('P', k=1), model_policy_args),
            'Model_Greedy_Oracle_k2_P': (dvfs.make_model_1_step_oracle_k('P', k=2), model_policy_args),
            'Model_Greedy_Oracle_k5_P': (dvfs.make_model_1_step_oracle_k('P', k=5), model_policy_args),
            'Model_Global_P':          (dvfs.make_model_global('P'),              model_policy_args),
        })
        # --- Reactive-axis decision variants (dampening / commitment) ---
        for w in [5, 10]:
            p_calls[f'Model_Greedy_Damp{w}_P'] = (
                dvfs.make_model_1_step_dampened('P', window=w), model_policy_args)
            p_calls[f'Model_Greedy_Commit{w}_P'] = (
                dvfs.make_model_1_step_commit('P', window=w), model_policy_args)
    if forecast_time_mat is not None:
        # Deployable workload-forecasting DVFS policy: greedy on the forecast tensor,
        # row i = causal forecast for chunk i (honest; no future PMU peeked).
        fc_p_args = (*policy_args, forecast_time_mat)
        p_calls['Model_Forecast_P'] = (dvfs.make_model_1_step_oracle('P'), fc_p_args)
        # --- Forecast-axis decision variants ---
        for w in [5, 10]:
            p_calls[f'Model_Forecast_Damp{w}_P'] = (
                dvfs.make_model_forecast_dampened('P', window=w), fc_p_args)
            p_calls[f'Model_Forecast_Commit{w}_P'] = (
                dvfs.make_model_forecast_commit('P', window=w), fc_p_args)

    # 2. E-Core DVFS Policies
    e_calls = {
        # --- Static baselines ---
        'Static_E_1.0GHz':       (dvfs.make_static('E_1.0GHz'),             policy_args),
        'Static_E_2.0GHz':       (dvfs.make_static('E_2.0GHz'),             policy_args),
        'Static_E_3.0GHz':       (dvfs.make_static('E_3.0GHz'),             policy_args),
        'Static_E_4.0GHz':       (dvfs.make_static('E_4.0GHz'),             policy_args),
        # --- Reactive heuristics (decide using prior chunk's proxy) ---
        'Performance_Gov_E':     (dvfs.make_performance_governor('E'),       policy_args),
        'Powersave_Gov_E':       (dvfs.make_powersave_governor('E'),         policy_args),
        'Ondemand_E':            (dvfs.make_ondemand('E'),                   policy_args),
        'Conservative_E':        (dvfs.make_conservative('E'),               policy_args),
        'Schedutil_PELT_E':      (dvfs.make_schedutil_pelt('E'),            policy_args),
        'Interactive_E':         (dvfs.make_interactive('E'),                policy_args),
        'Intel_HWP_E':           (dvfs.make_intel_hwp('E'),                  policy_args),
        'EWMA_E':                (dvfs.make_ewma_dvfs('E'),                  policy_args),
        'UCB1_E':                (dvfs.make_ucb1_dvfs('E'),                  policy_args),
        # --- Reactive oracle (repeats prior chunk's best config) ---
        'Reactive_Oracle_E':     (dvfs.make_reactive_oracle('E'),            policy_args),
        # --- Perfect-future heuristics (same algorithms, current chunk's true proxy) ---
        'Ondemand_Future_E':     (dvfs.make_ondemand('E',     temporal_mode='oracle_heuristic'), policy_args),
        'Conservative_Future_E': (dvfs.make_conservative('E', temporal_mode='oracle_heuristic'), policy_args),
        'Schedutil_Future_E':    (dvfs.make_schedutil_pelt('E', temporal_mode='oracle_heuristic'), policy_args),
        'Interactive_Future_E':  (dvfs.make_interactive('E', temporal_mode='oracle_heuristic'), policy_args),
        'Intel_HWP_Future_E':    (dvfs.make_intel_hwp('E',   temporal_mode='oracle_heuristic'), policy_args),
        'EWMA_Future_E':         (dvfs.make_ewma_dvfs('E',   temporal_mode='oracle_heuristic'), policy_args),
        'UCB1_Future_E':         (dvfs.make_ucb1_dvfs('E',   temporal_mode='oracle_heuristic'), policy_args),
        # --- Oracle bounds ---
        'Greedy_Oracle_E':       (dvfs.make_proactive_1_step('E'),           policy_args),
        'Global_Oracle_E':       (dvfs.make_global_viterbi('E'),             policy_args),
    }
    if e_model_time_mat is not None:
        e_model_policy_args = (*policy_args, e_model_time_mat)
        e_calls.update({
            # --- Reactive model (prior chunk PMU → predicted best E-core config) ---
            'Model_Greedy_E':          (dvfs.make_model_1_step('E'),              e_model_policy_args),
            # --- Perfect-future model (current chunk PMU, oracle temporal) ---
            'Model_Greedy_Oracle_E':   (dvfs.make_model_1_step_oracle('E'),       e_model_policy_args),
            # --- Oracle lookahead (average of next k+1 chunks' predictions) ---
            'Model_Greedy_Oracle_k1_E': (dvfs.make_model_1_step_oracle_k('E', k=1), e_model_policy_args),
            'Model_Greedy_Oracle_k2_E': (dvfs.make_model_1_step_oracle_k('E', k=2), e_model_policy_args),
            'Model_Greedy_Oracle_k5_E': (dvfs.make_model_1_step_oracle_k('E', k=5), e_model_policy_args),
            'Model_Global_E':          (dvfs.make_model_global('E'),              e_model_policy_args),
        })
        # --- Reactive-axis decision variants (dampening / commitment) ---
        for w in [5, 10]:
            e_calls[f'Model_Greedy_Damp{w}_E'] = (
                dvfs.make_model_1_step_dampened('E', window=w), e_model_policy_args)
            e_calls[f'Model_Greedy_Commit{w}_E'] = (
                dvfs.make_model_1_step_commit('E', window=w), e_model_policy_args)
    if e_forecast_time_mat is not None:
        fc_e_args = (*policy_args, e_forecast_time_mat)
        e_calls['Model_Forecast_E'] = (dvfs.make_model_1_step_oracle('E'), fc_e_args)
        # --- Forecast-axis decision variants ---
        for w in [5, 10]:
            e_calls[f'Model_Forecast_Damp{w}_E'] = (
                dvfs.make_model_forecast_dampened('E', window=w), fc_e_args)
            e_calls[f'Model_Forecast_Commit{w}_E'] = (
                dvfs.make_model_forecast_commit('E', window=w), fc_e_args)

    if fc_un_p_mat is not None:
        # Phase-UNAWARE forecast: identical policy, forecaster trained without phase
        # segmentation. Paired with Model_Forecast_P this isolates the scheduling value
        # of phase-awareness, which the standalone MAPE study cannot show.
        p_calls['Model_Forecast_Unaware_P'] = (dvfs.make_model_1_step_oracle('P'),
                                               (*policy_args, fc_un_p_mat))
    if fc_g_p_mat is not None:
        p_calls['Model_Forecast_Gated_P'] = (dvfs.make_model_1_step_oracle('P'),
                                             (*policy_args, fc_g_p_mat))
    if forecast_oracle_gated_mat is not None:
        p_calls['Forecast_Oracle_Gated_P'] = (dvfs.make_model_1_step_oracle('P'),
                                              (*policy_args, forecast_oracle_gated_mat))
    if forecast_oracle_mat is not None:
        fo_args = (*policy_args, forecast_oracle_mat)
        p_calls['Forecast_Oracle_P'] = (dvfs.make_model_1_step_oracle('P'), fo_args)
    if forecast_oracle_unaware_mat is not None:
        p_calls['Forecast_Oracle_Unaware_P'] = (dvfs.make_model_1_step_oracle('P'),
                                                (*policy_args, forecast_oracle_unaware_mat))

    if fc_un_e_mat is not None:
        e_calls['Model_Forecast_Unaware_E'] = (dvfs.make_model_1_step_oracle('E'),
                                               (*policy_args, fc_un_e_mat))
    if fc_g_e_mat is not None:
        e_calls['Model_Forecast_Gated_E'] = (dvfs.make_model_1_step_oracle('E'),
                                             (*policy_args, fc_g_e_mat))
    if forecast_oracle_gated_mat is not None:
        e_calls['Forecast_Oracle_Gated_E'] = (dvfs.make_model_1_step_oracle('E'),
                                              (*policy_args, forecast_oracle_gated_mat))
    if forecast_oracle_mat is not None:
        e_calls['Forecast_Oracle_E'] = (dvfs.make_model_1_step_oracle('E'),
                                        (*policy_args, forecast_oracle_mat))
    if forecast_oracle_unaware_mat is not None:
        e_calls['Forecast_Oracle_Unaware_E'] = (dvfs.make_model_1_step_oracle('E'),
                                                (*policy_args, forecast_oracle_unaware_mat))

    # 3. Heterogeneous Scheduling Policies (P vs E)
    hetero_calls = {
        'Proactive_Hetero_Oracle':    (sched.run_proactive_hetero_oracle, policy_args),
        'Greedy_Oracle_Hetero':       (sched.make_greedy_oracle_hetero(), policy_args),
        'IsoFreq_Oracle_1.0GHz':      (sched.make_global_oracle_fixed_freq('1.0GHz'), policy_args),
        'IsoFreq_Oracle_2.0GHz':      (sched.make_global_oracle_fixed_freq('2.0GHz'), policy_args),
        'IsoFreq_Oracle_3.0GHz':      (sched.make_global_oracle_fixed_freq('3.0GHz'), policy_args),
        'IsoFreq_Oracle_4.0GHz':      (sched.make_global_oracle_fixed_freq('4.0GHz'), policy_args),
        'IsoFreq_Reactive_Oracle_1.0GHz': (sched.make_reactive_oracle_fixed_freq('1.0GHz'), policy_args),
        'IsoFreq_Reactive_Oracle_2.0GHz': (sched.make_reactive_oracle_fixed_freq('2.0GHz'), policy_args),
        'IsoFreq_Reactive_Oracle_3.0GHz': (sched.make_reactive_oracle_fixed_freq('3.0GHz'), policy_args),
        'IsoFreq_Reactive_Oracle_4.0GHz': (sched.make_reactive_oracle_fixed_freq('4.0GHz'), policy_args),
        'IsoFreq_Oracle_Heuristic_1.0GHz': (sched.make_isofreq_oracle_heuristic('1.0GHz'), policy_args),
        'IsoFreq_Oracle_Heuristic_2.0GHz': (sched.make_isofreq_oracle_heuristic('2.0GHz'), policy_args),
        'IsoFreq_Oracle_Heuristic_3.0GHz': (sched.make_isofreq_oracle_heuristic('3.0GHz'), policy_args),
        'IsoFreq_Oracle_Heuristic_4.0GHz': (sched.make_isofreq_oracle_heuristic('4.0GHz'), policy_args),
        'EAS_Hetero':          (sched.make_eas_hetero(),          policy_args),
        'EAS_With_DVFS':       (sched.make_eas_with_dvfs(),       policy_args),
        'Threshold_Migration': (sched.make_threshold_migration(), policy_args),
        'Thread_Director':     (sched.make_thread_director(),     policy_args),
        'Micro_EAS':           (sched.run_micro_eas,              policy_args),
        'UCB1_Hetero':         (sched.make_ucb1_hetero(),         policy_args),
        'EAS_Oracle_Hetero':   (sched.make_eas_oracle(),          policy_args),
        'Thread_Director_Oracle': (sched.make_thread_director_oracle(), policy_args),
    }
    if cross_proc_time_mat is not None:
        cross_proc_args = (*policy_args, cross_proc_time_mat)
        for freq in ['1.0GHz', '2.0GHz', '3.0GHz', '4.0GHz']:
            hetero_calls[f'Model_IsoFreq_{freq}']        = (sched.make_isofreq_model(freq),        cross_proc_args)
            hetero_calls[f'IsoFreq_Model_Oracle_{freq}'] = (sched.make_isofreq_model_oracle(freq), cross_proc_args)
            for k in [1, 2, 5]:
                hetero_calls[f'IsoFreq_Model_Oracle_k{k}_{freq}'] = (sched.make_isofreq_model_oracle_k(freq, k=k), cross_proc_args)
            for h in [5, 10, 20, 50, 100]:
                hetero_calls[f'IsoFreq_Model_MPC_Oracle_W{h}_{freq}'] = (sched.make_isofreq_model_mpc_oracle(freq, horizon=h), cross_proc_args)
            for w in [5, 10]:
                hetero_calls[f'Model_IsoFreq_Damp{w}_{freq}'] = (sched.make_isofreq_model_dampened(freq, window=w), cross_proc_args)
            for w in [5, 10]:
                hetero_calls[f'Model_IsoFreq_Commit{w}_{freq}'] = (sched.make_isofreq_model_commit(freq, window=w), cross_proc_args)
    if cross_proc_forecast_time_mat is not None:
        # Deployable cross-proc (migration) FORECAST: row i = causal forecast of the target-core
        # config from current-core history. make_isofreq_model_oracle reads row i (honest here).
        cp_fc_args = (*policy_args, cross_proc_forecast_time_mat)
        for freq in ['1.0GHz', '2.0GHz', '3.0GHz', '4.0GHz']:
            hetero_calls[f'Model_IsoFreq_Forecast_{freq}'] = (sched.make_isofreq_model_oracle(freq), cp_fc_args)
            for w in [5, 10]:
                hetero_calls[f'Model_IsoFreq_Forecast_Damp{w}_{freq}'] = (
                    sched.make_isofreq_model_forecast_dampened(freq, window=w), cp_fc_args)
            for w in [5, 10]:
                hetero_calls[f'Model_IsoFreq_Forecast_Commit{w}_{freq}'] = (
                    sched.make_isofreq_model_forecast_commit(freq, window=w), cp_fc_args)
    if full_model_time_mat is not None:
        full_model_args = (*policy_args, full_model_time_mat)
        hetero_calls['Model_Reactive_Hetero']        = (sched.make_hetero_model_reactive(), full_model_args)
        hetero_calls['Model_Greedy_Oracle_Hetero']   = (sched.make_hetero_model_oracle(),   full_model_args)
        for k in [1, 2, 5]:
            hetero_calls[f'Model_Greedy_Oracle_k{k}_Hetero'] = (sched.make_hetero_model_oracle_k(k=k), full_model_args)
        # Reactive-axis decision variants. These read the MODEL tensor, so they belong
        # here and not under the forecast guard.
        for w in [5, 10]:
            hetero_calls[f'Model_Reactive_Damp{w}_Hetero'] = (sched.make_hetero_model_dampened(window=w), full_model_args)
            hetero_calls[f'Model_Reactive_Commit{w}_Hetero'] = (sched.make_hetero_model_commit(window=w), full_model_args)
        for h in [5, 10]:
            hetero_calls[f'Model_MPC_Oracle_W{h}_Hetero'] = (sched.make_hetero_model_mpc_oracle(horizon=h), full_model_args)
    # --- 2x2: which model is responsible for the heterogeneous deficit? ---
    # Substitute ground truth on one prediction axis at a time. Reactive temporal
    # throughout, so the only thing varying is which model supplies which entries.
    #   Oracle x Oracle : the reactive-oracle bound (no model error at all)
    #   Model  x Oracle : cross-frequency model error only
    #   Oracle x Model  : cross-processor model error only
    #   Model  x Model  : == Model_Reactive_Hetero (both), for reference
    if model_time_mat is not None and cross_proc_time_mat is not None:
        # build_oracle_axis_mat returns all 8 source rows (ALL_MODEL_CONFIGS order), but
        # _load_full_model_time_mat consumes the cross-frequency tensors per core: a
        # 4-row P-source tensor and a 4-row E-source tensor. Slice accordingly. Passing
        # the 8-row tensor straight through hands the P-source rows the E-source data,
        # which leaves every P->P frequency entry at 1e6 and silently removes P-core
        # DVFS from the action space.
        _cf_oracle = build_oracle_axis_mat(time_mat, configs, min_len, 'cross_freq')
        _cp_oracle = build_oracle_axis_mat(time_mat, configs, min_len, 'cross_proc')
        _cf_oracle_p, _cf_oracle_e = _cf_oracle[4:8], _cf_oracle[0:4]
        _cf_model = (_load_full_model_time_mat(model_time_mat, _cp_oracle, e_model_time_mat)
                     if e_model_time_mat is not None else None)
        for tag, mat in [
            ('OracleCF_OracleCP',
             _load_full_model_time_mat(_cf_oracle_p, _cp_oracle, _cf_oracle_e)),
            ('ModelCF_OracleCP', _cf_model),
            ('OracleCF_ModelCP',
             _load_full_model_time_mat(_cf_oracle_p, cross_proc_time_mat, _cf_oracle_e)),
        ]:
            if mat is None:
                continue
            hetero_calls[f'Model_Reactive_Hetero_{tag}'] = (
                sched.make_hetero_model_reactive(), (*policy_args, mat))
            hetero_calls[f'Model_Oracle_Hetero_{tag}'] = (
                sched.make_hetero_model_oracle(), (*policy_args, mat))

    if full_forecast_time_mat is not None:
        # Deployable FORECAST over the full core x freq space: row i = causal forecast.
        ff_args = (*policy_args, full_forecast_time_mat)
        hetero_calls['Model_Forecast_Hetero'] = (sched.make_hetero_model_oracle(), ff_args)
        for w in [5, 10]:
            hetero_calls[f'Model_Forecast_Damp{w}_Hetero'] = (
                sched.make_hetero_model_forecast_dampened(window=w), ff_args)
            hetero_calls[f'Model_Forecast_Commit{w}_Hetero'] = (
                sched.make_hetero_model_forecast_commit(window=w), ff_args)

    if fc_un_full_mat is not None:
        hetero_calls['Model_Forecast_Unaware_Hetero'] = (sched.make_hetero_model_oracle(),
                                                         (*policy_args, fc_un_full_mat))
    if fc_g_full_mat is not None:
        hetero_calls['Model_Forecast_Gated_Hetero'] = (sched.make_hetero_model_oracle(),
                                                       (*policy_args, fc_g_full_mat))
        for w in [5, 10]:
            hetero_calls[f'Model_Forecast_Gated_Damp{w}_Hetero'] = (
                sched.make_hetero_model_forecast_dampened(window=w), (*policy_args, fc_g_full_mat))
    if forecast_oracle_gated_mat is not None:
        hetero_calls['Forecast_Oracle_Gated_Hetero'] = (
            sched.make_hetero_model_oracle(), (*policy_args, forecast_oracle_gated_mat))
    if forecast_oracle_mat is not None:
        hetero_calls['Forecast_Oracle_Hetero'] = (sched.make_hetero_model_oracle(),
                                                  (*policy_args, forecast_oracle_mat))
    if forecast_oracle_unaware_mat is not None:
        hetero_calls['Forecast_Oracle_Unaware_Hetero'] = (
            sched.make_hetero_model_oracle(), (*policy_args, forecast_oracle_unaware_mat))

    # 4. Combined DVFS + Migration Policies
    combined_calls = {
        'Reactive_Combined_W1':   (comb.make_reactive_n_step_combined(lookback=1),  policy_args),
        'Reactive_Combined_W5':   (comb.make_reactive_n_step_combined(lookback=5),  policy_args),
        'Reactive_Combined_W10':  (comb.make_reactive_n_step_combined(lookback=10), policy_args),
        'MPC_Oracle_Combined_W1': (comb.make_proactive_n_step_combined(horizon=1),  policy_args),
        'MPC_Oracle_Combined_W5': (comb.make_proactive_n_step_combined(horizon=5),  policy_args),
        'MPC_Oracle_Combined_W10':(comb.make_proactive_n_step_combined(horizon=10), policy_args),
    }

    def _trace_from_path(actions, local_names, m):
        """Trace for a known action path, with the warmup penalty applied if enabled.

        Split out of _apply_warmup_trace so a cached path can be re-scored without
        re-running the (expensive) full-trace DP.
        """
        idx = [configs.index(c) for c in local_names]
        t_sub = time_mat[:, idx]
        e_sub = energy_mat[:, idx]
        lat_sub = trans_lat[np.ix_(idx, idx)]
        nrg_sub = trans_nrg[np.ix_(idx, idx)]
        if apply_warmup and not (WARMUP_A_PtoE == 0.0 and WARMUP_A_EtoP == 0.0):
            t_sub = warmup_model.apply_warmup_penalty(
                t_sub, actions, local_names,
                WARMUP_A_PtoE, WARMUP_TAU_PtoE,
                WARMUP_A_EtoP, WARMUP_TAU_EtoP, WARMUP_K,
            )
        return accumulate_trace(t_sub, e_sub, lat_sub, nrg_sub, actions, metric=m)

    def _apply_warmup_trace(fn, args_, m):
        """Run policy, apply cache-warmup penalty to cross-cluster migrations, recompute trace."""
        trace, actions, local_names = fn(*args_, _return_actions=True, metric=m)
        st = compute_trace_stats(list(actions), local_names)
        if WARMUP_A_PtoE == 0.0 and WARMUP_A_EtoP == 0.0:
            return trace, st
        # Reconstruct the submatrix for this policy's action space from the full matrices.
        # args_[0]=time_mat, args_[1]=energy_mat, args_[3]=configs, args_[5]=trans_lat, args_[6]=trans_nrg
        idx = [configs.index(c) for c in local_names]
        t_sub   = time_mat[:, idx]
        e_sub   = energy_mat[:, idx]
        lat_sub = trans_lat[np.ix_(idx, idx)]
        nrg_sub = trans_nrg[np.ix_(idx, idx)]
        pen_t = warmup_model.apply_warmup_penalty(
            t_sub, actions, local_names,
            WARMUP_A_PtoE, WARMUP_TAU_PtoE,
            WARMUP_A_EtoP, WARMUP_TAU_EtoP, WARMUP_K,
        )
        trace = accumulate_trace(pen_t, e_sub, lat_sub, nrg_sub, actions, metric=m)
        return trace, st

    def _run_calls(calls, traces_by_metric):
        """Evaluate all policies, using dual-metric fast path for metric-independent ones."""
        for name, (fn, args_) in calls.items():
            if getattr(fn, 'metric_independent', False):
                # Dual-metric fast path: one action sequence, scored under both metrics.
                # Warmup must still be charged -- the heterogeneous heuristics are
                # metric-independent AND migrate, so skipping it here scored them with
                # no cache-warmup penalty while every model policy paid one.
                traces, actions, local_names = fn(*args_, metrics=METRICS, _return_actions=True)
                st = compute_trace_stats(list(actions), local_names)
                for m in METRICS:
                    traces_by_metric[m][name] = (_trace_from_path(actions, local_names, m)
                                                 if apply_warmup else traces[m])
                    _record_diag(diag_results, wl, ph, m, name, st)
            elif getattr(fn, 'is_viterbi_oracle', False):
                if os.environ.get('SKIP_VITERBI'):
                    continue   # skip the expensive full-trace DP oracles for a fast ladder
                # Full-trace DP: the expensive path. Cache the ACTION PATH rather than
                # the trace, so the cache is reusable when --apply_warmup is on (the DP
                # is warmup-unaware by design, so the path is identical either way and
                # only the re-scoring differs). Caching the trace instead made the cache
                # dead for every warmup run, which is every run in the chapter.
                #
                # Model-based Viterbi (Model_Global_*) depends on the prediction tensor,
                # so its key carries the prediction-set tag. A true oracle's path depends
                # only on the traces and is shared across prediction sets.
                # The DP optimizes E*T^p over the traces, so its path depends on the
                # trace set AND on --power_mode (which defines energy_mat). Neither was
                # in the key. Model-based Viterbi additionally depends on the prediction
                # set. Warmup is NOT in the key: the DP is warmup-unaware by design, so
                # the path is identical either way and only the re-scoring differs.
                tag = env_tag if not getattr(fn, 'uses_model', False) else f'{env_tag}-{pred_tag}'
                for m in METRICS:
                    cache_file = (viterbi_cache_dir / f"{wl}__{ph}__{name}__{tag}__{m}.npz"
                                  if viterbi_cache_dir is not None else None)
                    actions = local_names = None
                    if cache_file is not None and cache_file.exists():
                        try:
                            z = np.load(cache_file, allow_pickle=False)
                            actions, local_names = z['actions'], [str(x) for x in z['names']]
                        except Exception:
                            actions = local_names = None
                    if actions is None:
                        _tr, actions, local_names = fn(*args_, _return_actions=True, metric=m)
                        if cache_file is not None:
                            np.savez(cache_file, actions=np.asarray(actions),
                                     names=np.array(local_names, dtype='U16'))
                    st = compute_trace_stats(list(actions), local_names)
                    traces_by_metric[m][name] = _trace_from_path(actions, local_names, m)
                    _record_diag(diag_results, wl, ph, m, name, st)
            elif apply_warmup and getattr(fn, 'returns_actions', False):
                for m in METRICS:
                    tr, st = _apply_warmup_trace(fn, args_, m)
                    traces_by_metric[m][name] = tr
                    _record_diag(diag_results, wl, ph, m, name, st)
            else:
                for m in METRICS:
                    tr, st = _call_with_stats(fn, *args_, metric=m)
                    traces_by_metric[m][name] = tr
                    _record_diag(diag_results, wl, ph, m, name, st)

    p_traces   = {m: {} for m in METRICS}
    e_traces   = {m: {} for m in METRICS}
    hetero_traces   = {m: {} for m in METRICS}
    combined_traces = {m: {} for m in METRICS}

    _run_calls(p_calls,       p_traces)
    _run_calls(e_calls,       e_traces)
    _run_calls(hetero_calls,  hetero_traces)
    _run_calls(combined_calls, combined_traces)

    # Mirror shared industry policies into combined_traces
    for key in ('EAS_Hetero', 'EAS_With_DVFS', 'Threshold_Migration', 'Thread_Director',
                'UCB1_Hetero', 'Proactive_Hetero_Oracle'):
        for m in METRICS:
            if key in hetero_traces[m]:
                combined_traces[m][key] = hetero_traces[m][key]

    # Aggregate Results
    for m_type in METRICS:
        all_traces = {**p_traces[m_type], **e_traces[m_type],
                      **hetero_traces[m_type], **combined_traces[m_type]}
        for name, tr in all_traces.items():
            if len(tr) > 0:
                summary_results.append({'Workload': wl, 'Phase': ph, 'Metric': m_type, 'Policy': name, 'Final_Value': tr[-1]})

    return summary_results, diag_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--exclude_workloads', type=str, default='',
                        help="Comma-separated workload names to skip entirely. Use to drop "
                             "workloads with known-incomplete prediction sets so the rest of "
                             "the run can keep --strict_predictions on rather than silently "
                             "falling back to oracle times for them.")
    parser.add_argument('--limit', type=int, default=0,
                        help="Process at most N workload-phases. For smoke tests: a full "
                             "run is 147 phases and takes hours, so --limit 6 exercises "
                             "every code path in minutes.")
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--power_mode', type=str, default='per_sample', choices=['per_sample', 'baseline'],
                         help="'per_sample' uses measured per-chunk power (default); "
                              "'baseline' uses the fixed get_power_w lookup table for all configs.")
    parser.add_argument('--cross_freq_p_pred_dir', type=str, default=None,
                        help="Directory with precomputed P-core cross-freq model predictions "
                             "(output of cross_freq_precompute.py). When provided, adds "
                             "Model_Greedy_P, Model_Greedy_Oracle_P, Model_Global_P policies.")
    parser.add_argument('--cross_freq_e_pred_dir', type=str, default=None,
                        help="Directory with precomputed E-core cross-freq model predictions "
                             "(output of cross_freq_precompute.py --core_type E). When provided, adds "
                             "Model_Greedy_E, Model_Greedy_Oracle_E, Model_Global_E policies.")
    parser.add_argument('--strict_predictions', action='store_true',
                        help="Abort if any prediction file is missing instead of silently "
                             "substituting oracle times. Use whenever a prediction dir is "
                             "passed and expected to be complete: the fallback otherwise "
                             "fabricates a model result that looks like a real one.")
    parser.add_argument('--cross_freq_p_forecast_dir', type=str, default=None,
                        help="Directory with WALK-FORWARD forecast predictions (dump_dvfs_forecast.py), "
                             "P-core. Each row i is a causal forecast for chunk i. Adds Model_Forecast_P.")
    parser.add_argument('--cross_freq_e_forecast_dir', type=str, default=None,
                        help="Directory with WALK-FORWARD forecast predictions (dump_dvfs_forecast.py), "
                             "E-core. Adds Model_Forecast_E.")
    parser.add_argument('--cross_freq_p_forecast_gated_dir', type=str, default=None,
                        help="GATED P-core cross-freq forecasts (dump with --gate persist): the "
                             "forecaster defers to persistence per chunk unless it has been "
                             "beating it on recent history. Adds Model_Forecast_Gated_P.")
    parser.add_argument('--cross_freq_e_forecast_gated_dir', type=str, default=None,
                        help="GATED E-core cross-freq forecasts. Adds Model_Forecast_Gated_E.")
    parser.add_argument('--cross_proc_forecast_gated_dir', type=str, default=None,
                        help="GATED cross-processor forecasts. Adds Model_Forecast_Gated_Hetero.")
    parser.add_argument('--forecast_oracle_gated_dir', type=str, default=None,
                        help="GATED self-forecasts. Adds Forecast_Oracle_Gated_*.")
    parser.add_argument('--cross_freq_p_forecast_unaware_dir', type=str, default=None,
                        help="Phase-UNAWARE P-core cross-freq forecasts (dump with "
                             "--method global). Adds Model_Forecast_Unaware_P.")
    parser.add_argument('--cross_freq_e_forecast_unaware_dir', type=str, default=None,
                        help="Phase-UNAWARE E-core cross-freq forecasts. Adds "
                             "Model_Forecast_Unaware_E.")
    parser.add_argument('--cross_proc_forecast_unaware_dir', type=str, default=None,
                        help="Phase-UNAWARE cross-processor forecasts. With the two "
                             "cross-freq unaware dirs, adds Model_Forecast_Unaware_Hetero.")
    parser.add_argument('--forecast_oracle_unaware_dir', type=str, default=None,
                        help="Phase-UNAWARE self-forecasts. Adds Forecast_Oracle_Unaware_*.")
    parser.add_argument('--forecast_oracle_dir', type=str, default=None,
                        help="Directory of self-forecasts (dump_dvfs_forecast.py --mode "
                             "self_forecast): {config}/{wl}_phase{n}.csv with Time_pred_{config}. "
                             "Adds Forecast_Oracle_{P,E,Hetero} -- the forecast-with-perfect-"
                             "translation bound, which isolates forecast error from model error.")
    parser.add_argument('--cross_proc_forecast_dir', type=str, default=None,
                        help="Directory with WALK-FORWARD cross-proc forecast predictions "
                             "(dump_dvfs_forecast.py --target_core). Adds Model_IsoFreq_Forecast_{freq}.")
    parser.add_argument('--cross_proc_pred_dir', type=str, default=None,
                        help="Directory with precomputed cross-proc model predictions "
                             "(output of cross_proc_precompute.py). When provided, adds "
                             "Model_IsoFreq_{1-4}GHz policies.")
    parser.add_argument('--viterbi_cache_dir', type=str, default=None,
                        help="Directory to cache/load global-oracle Viterbi traces. "
                             "On first run, computed traces are saved here; subsequent "
                             "runs with the same data load from cache, skipping the DP.")
    parser.add_argument('--apply_warmup', action='store_true', default=False,
                        help="Apply cache-warmup time penalty after P↔E migrations. "
                             "Uses WARMUP_A/TAU/K constants defined in main.py. "
                             "Only affects policies that return an action sequence; "
                             "Viterbi oracle baselines are evaluated without warmup.")
    parser.add_argument('--cross_phase', action='store_true', default=False,
                        help="Concatenate all phases per workload into one long trace "
                             "before simulation. Phase-transition chunks are then visible "
                             "in-band, exposing the reactive vs oracle gap at transitions. "
                             "Results are labeled Phase='all'. Use a separate --output_dir "
                             "to avoid mixing with per-phase results.")
    parser.add_argument('--arch', type=str, default='x86', choices=['x86', 'arm_edge'],
                        help="Target architecture: 'x86' (P/E cores) or 'arm_edge' (L/B cores)")
    args = parser.parse_args()

    # Missing prediction files silently become oracle times, which makes a model
    # policy indistinguishable from its oracle. Count them always; abort on them
    # when the caller asserts the prediction set should be complete.
    data_loader.STRICT_PREDICTIONS = args.strict_predictions
    data_loader.reset_fallback_counts()

    import pandas as pd

    # Identity of the prediction set feeding the model policies. Model-based Viterbi
    # results depend on it, so it keys their cache entries: without this, running cap5
    # against a cache populated by the raw run would silently return raw's Model_Global_*.
    _pred_srcs = [args.cross_freq_p_pred_dir, args.cross_freq_e_pred_dir,
                  args.cross_proc_pred_dir, args.cross_freq_p_forecast_dir,
                  args.cross_freq_e_forecast_dir, args.cross_proc_forecast_dir]
    pred_tag = hashlib.md5('|'.join(str(x) for x in _pred_srcs).encode()).hexdigest()[:8]
    # Everything a TRUE-oracle Viterbi path depends on: the traces it reads and the
    # power model that turns them into energy.
    # Bump VITERBI_CACHE_VERSION whenever the DP or the start convention changes:
    # env_tag covers the data a path depends on, not the code that produced it, so a
    # stale cache would otherwise silently return paths from the previous semantics.
    env_tag = hashlib.md5(
        f'{args.input_dir}|{args.power_mode}|v{VITERBI_CACHE_VERSION}'.encode()).hexdigest()[:8]

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    cross_freq_p_pred_dir = Path(args.cross_freq_p_pred_dir) if args.cross_freq_p_pred_dir else None
    cross_freq_e_pred_dir = Path(args.cross_freq_e_pred_dir) if args.cross_freq_e_pred_dir else None
    cross_proc_pred_dir = Path(args.cross_proc_pred_dir) if args.cross_proc_pred_dir else None
    cross_freq_p_forecast_unaware_dir = Path(args.cross_freq_p_forecast_unaware_dir) if args.cross_freq_p_forecast_unaware_dir else None
    cross_freq_e_forecast_unaware_dir = Path(args.cross_freq_e_forecast_unaware_dir) if args.cross_freq_e_forecast_unaware_dir else None
    cross_proc_forecast_unaware_dir = Path(args.cross_proc_forecast_unaware_dir) if args.cross_proc_forecast_unaware_dir else None
    forecast_oracle_unaware_dir = Path(args.forecast_oracle_unaware_dir) if args.forecast_oracle_unaware_dir else None
    cross_freq_p_forecast_gated_dir = Path(args.cross_freq_p_forecast_gated_dir) if args.cross_freq_p_forecast_gated_dir else None
    cross_freq_e_forecast_gated_dir = Path(args.cross_freq_e_forecast_gated_dir) if args.cross_freq_e_forecast_gated_dir else None
    cross_proc_forecast_gated_dir = Path(args.cross_proc_forecast_gated_dir) if args.cross_proc_forecast_gated_dir else None
    forecast_oracle_gated_dir = Path(args.forecast_oracle_gated_dir) if args.forecast_oracle_gated_dir else None
    forecast_oracle_dir = Path(args.forecast_oracle_dir) if args.forecast_oracle_dir else None
    cross_freq_p_forecast_dir = Path(args.cross_freq_p_forecast_dir) if args.cross_freq_p_forecast_dir else None
    cross_freq_e_forecast_dir = Path(args.cross_freq_e_forecast_dir) if args.cross_freq_e_forecast_dir else None
    cross_proc_forecast_dir = Path(args.cross_proc_forecast_dir) if args.cross_proc_forecast_dir else None
    viterbi_cache_dir = Path(args.viterbi_cache_dir) if args.viterbi_cache_dir else None
    if viterbi_cache_dir is not None:
        viterbi_cache_dir.mkdir(parents=True, exist_ok=True)

    if args.arch == 'arm_edge':
        pattern = re.compile(r"speedups_([LB]_[0-9.]+GHz)_(.+)_phase(\d+)\.csv")
        configs = ['L_1.0GHz', 'B_1.0GHz']
    else:
        pattern = re.compile(r"speedups_([PE]_[0-9.]+GHz)_(.+)_phase(\d+)\.csv")
        configs = ['E_1.0GHz', 'E_2.0GHz', 'E_3.0GHz', 'E_4.0GHz',
                   'P_1.0GHz', 'P_2.0GHz', 'P_3.0GHz', 'P_4.0GHz']
    pairs = set(m.groups()[1:] for f in input_path.glob("speedups_*.csv") if (m := pattern.search(f.name)))
    if args.exclude_workloads:
        drop = {w.strip() for w in args.exclude_workloads.split(',') if w.strip()}
        before = len(pairs)
        pairs = {(wl, ph) for wl, ph in pairs if wl not in drop}
        print(f"--exclude_workloads: dropped {before - len(pairs)} phases "
              f"from {len(drop)} workloads -> {len(pairs)} remain")
    if args.limit:
        pairs = set(sorted(pairs)[:args.limit])
        print(f"--limit {args.limit}: processing {len(pairs)} workload-phases (smoke test)")

    common_kwargs = dict(
        power_mode=args.power_mode,
        cross_freq_p_pred_dir=cross_freq_p_pred_dir,
        cross_freq_e_pred_dir=cross_freq_e_pred_dir,
        cross_proc_pred_dir=cross_proc_pred_dir,
        cross_freq_p_forecast_dir=cross_freq_p_forecast_dir,
        cross_freq_e_forecast_dir=cross_freq_e_forecast_dir,
        cross_proc_forecast_dir=cross_proc_forecast_dir,
        forecast_oracle_dir=forecast_oracle_dir,
        cross_freq_p_forecast_unaware_dir=cross_freq_p_forecast_unaware_dir,
        cross_freq_e_forecast_unaware_dir=cross_freq_e_forecast_unaware_dir,
        cross_proc_forecast_unaware_dir=cross_proc_forecast_unaware_dir,
        forecast_oracle_unaware_dir=forecast_oracle_unaware_dir,
        cross_freq_p_forecast_gated_dir=cross_freq_p_forecast_gated_dir,
        cross_freq_e_forecast_gated_dir=cross_freq_e_forecast_gated_dir,
        cross_proc_forecast_gated_dir=cross_proc_forecast_gated_dir,
        forecast_oracle_gated_dir=forecast_oracle_gated_dir,
        viterbi_cache_dir=viterbi_cache_dir,
        apply_warmup=args.apply_warmup,
        pred_tag=pred_tag,
        env_tag=env_tag,
    )

    if args.cross_phase:
        from collections import defaultdict
        wl_to_phases = defaultdict(list)
        for wl, ph in pairs:
            wl_to_phases[wl].append(int(ph))
        print(f"Starting Modular Simulator (cross-phase) for {len(wl_to_phases)} workloads...")
        submit_items = [
            (wl, 'all', sorted(ph_list))
            for wl, ph_list in wl_to_phases.items()
        ]
    else:
        print(f"Starting Modular Simulator for {len(pairs)} workload-phases...")
        submit_items = [(wl, ph, None) for wl, ph in pairs]

    all_summary = []
    all_diag = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(
                process_workload, wl, ph, pairs, input_path, configs,
                **common_kwargs, phases=phases
            ): (wl, ph)
            for wl, ph, phases in submit_items
        }
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res:
                summary, diag = res
                all_summary.extend(summary)
                all_diag.extend(diag)

    if all_summary:
        df = pd.DataFrame(all_summary)
        print(f"\nSimulation complete. Generating plots and CSVs...")
        _fb = data_loader.fallback_report()
        if _fb:
            print("\n" + "=" * 78)
            print(_fb)
            print("=" * 78 + "\n")
        plotter.generate_all_plots(df, output_path)
        if all_diag:
            diag_df = pd.DataFrame(all_diag)
            diag_path = output_path / 'diagnostics.csv'
            diag_df.to_csv(diag_path, index=False)
            print(f"Diagnostics written to {diag_path}")
    else:
        print("\nNo valid data processed.")

if __name__ == "__main__":
    main()
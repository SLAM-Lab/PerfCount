import argparse
import re
import numpy as np
from pathlib import Path
import concurrent.futures

# Direct local imports (since they are in the same folder as main.py)
from data_loader import load_phase_data
import dvfs_policies as dvfs
import scheduling_policies as sched
import plotter
import combined_policies as comb
import warmup_model

# ==========================================
# GLOBAL HARDWARE DEFINES
# ==========================================
# Cache warm-up penalty parameters (fill in from warmup_params.csv after measurement).
# Set both A values to 0.0 to verify results are identical to the no-penalty baseline.
# To apply: call warmup_model.apply_warmup_penalty(time_mat, policy_path, configs, ...)
# for each policy after extracting its per-chunk config path.
WARMUP_A_PtoE   = 0.0   # amplitude:  P→E warm-up slowdown (measured via warmup_collection.sh)
WARMUP_TAU_PtoE = 1.0   # decay time: P→E (chunks until steady-state)
WARMUP_A_EtoP   = 0.0   # amplitude:  E→P warm-up slowdown
WARMUP_TAU_EtoP = 1.0   # decay time: E→P (chunks)
WARMUP_K        = 50    # number of post-migration chunks to penalize

# P↔E context switch: mean 4.47μs, symmetric within 0.1μs (ctx_switch_bench, 10 reps × 5000 migrations)
MIG_LAT_S  = 4.47e-6
# Migration energy: placeholder estimate (4.5μs × ~2W core power @ 3GHz).
# RAPL-based direct measurement was attempted (continuous-spin sweep,
# duty-cycled bench, same-core control — see power_collection/ctx_switch/
# ctx_switch_power_freq_sweep.sh, ctx_switch_power_duty_test.sh,
# ctx_switch_power_duty_control.sh + analyze_*.py). A single migration's
# energy budget (~4.47μs × O(1-2W) ≈ O(1-10nJ)) is below RAPL's ~50ms /
# ~10-50mW resolution; measured "energy per migration" varied by orders of
# magnitude (and sign) with duty cycle / loop rate, confirming those values
# are measurement artifacts, not signal. 9e-9 remains the best available
# order-of-magnitude placeholder.
MIG_NRG_J  = 9e-9
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

def get_dvfs_cost(start_cfg, end_cfg):
    """Calculates true Latency (s) and Energy (J) for a specific frequency hop."""
    s_type, s_freq_str = start_cfg.split('_')
    e_type, e_freq_str = end_cfg.split('_')
    
    s_freq = float(s_freq_str.replace('GHz', ''))
    e_freq = float(e_freq_str.replace('GHz', ''))
    
    if s_freq == e_freq:
        return 0.0, 0.0
        
    # Clamp bounds to 4.0GHz for safety (prevents crashes on P_5.0GHz evaluation)
    s_freq, e_freq = min(s_freq, 4.0), min(e_freq, 4.0)
        
    lat_map = P_LAT_US if s_type == 'P' else E_LAT_US
    nrg_map = P_NRG_J if s_type == 'P' else E_NRG_J
    
    # Fallback to defaults if a missing transition is requested
    lat_us = lat_map.get((s_freq, e_freq), DVFS_LAT_S * 1e6)
    nrg_j = nrg_map.get((s_freq, e_freq), DVFS_NRG_J)
    
    lat_s = lat_us * 1e-6
    
    return lat_s, nrg_j

def build_transition_matrices(configs):
    n = len(configs)
    lat_mat, nrg_mat = np.zeros((n, n)), np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j: continue
            
            # Cross-cluster Migration Cost
            if configs[i][0] != configs[j][0]:
                lat_mat[i, j], nrg_mat[i, j] = MIG_LAT_S, MIG_NRG_J
            # Same-cluster DVFS Cost
            else:
                lat_s, nrg_j = get_dvfs_cost(configs[i], configs[j])
                lat_mat[i, j], nrg_mat[i, j] = lat_s, nrg_j
                    
    return lat_mat, nrg_mat

def process_workload(wl, ph, pairs, input_path, bar_dir, trace_dir, configs, power_mode='per_sample'):
    data = load_phase_data(wl, ph, input_path, configs, power_mode=power_mode)
    if data is None: return None
    
    time_mat, energy_mat, proxy_signal, valid_configs, min_len = data
    min_len = len(time_mat)
    trans_lat, trans_nrg = build_transition_matrices(configs)
    
    summary_results = []
    
    # Define common arguments for policy calls
    policy_args = (time_mat, energy_mat, proxy_signal, configs, valid_configs, trans_lat, trans_nrg)
    
    metrics = ['EDP', 'ED2P']
    for m_type in metrics:
        
        # 1. P-Core DVFS Policies
        p_traces = {
            # Baselines: static pinned frequencies
            'Static_P_1.0GHz': dvfs.make_static('P_1.0GHz')(*policy_args, metric=m_type),
            'Static_P_2.0GHz': dvfs.make_static('P_2.0GHz')(*policy_args, metric=m_type),
            'Static_P_3.0GHz': dvfs.make_static('P_3.0GHz')(*policy_args, metric=m_type),
            'Static_P_4.0GHz': dvfs.make_static('P_4.0GHz')(*policy_args, metric=m_type),
            # Reactive: uses only past chunk data (causal, practical)
            'Reactive_1_Step_P': dvfs.make_reactive_1_step('P')(*policy_args, metric=m_type),
            # Oracle policies: have access to current/future timing data (not practically deployable)
            'Greedy_Oracle_P': dvfs.make_proactive_1_step('P')(*policy_args, metric=m_type),
            'MPC_Oracle_P_W5': dvfs.make_proactive_n_step('P', horizon=5)(*policy_args, metric=m_type),
            'MPC_Oracle_P_W10': dvfs.make_proactive_n_step('P', horizon=10)(*policy_args, metric=m_type),
            'Global_Oracle_P': dvfs.make_global_viterbi('P')(*policy_args, metric=m_type),
            # Industry OS governors (use proxy signal, practical)
            'Performance_Gov_P': dvfs.run_performance_governor(*policy_args, metric=m_type),
            'Ondemand_P': dvfs.make_ondemand('P')(*policy_args, metric=m_type),
            'Conservative_P': dvfs.make_conservative('P')(*policy_args, metric=m_type),
            'Schedutil_PELT_P': dvfs.make_schedutil_pelt('P')(*policy_args, metric=m_type),
            'Intel_HWP_P': dvfs.run_intel_hwp(*policy_args, metric=m_type),
            # Academic prediction / online learning (practical)
            'EWMA_P': dvfs.make_ewma_dvfs('P')(*policy_args, metric=m_type),
            'UCB1_P': dvfs.make_ucb1_dvfs('P')(*policy_args, metric=m_type),
            'Random_P': dvfs.make_random_dvfs('P')(*policy_args, metric=m_type),
        }

        # 2. E-Core DVFS Policies
        e_traces = {
            # Baselines: static pinned frequencies
            'Static_E_1.0GHz': dvfs.make_static('E_1.0GHz')(*policy_args, metric=m_type),
            'Static_E_2.0GHz': dvfs.make_static('E_2.0GHz')(*policy_args, metric=m_type),
            'Static_E_3.0GHz': dvfs.make_static('E_3.0GHz')(*policy_args, metric=m_type),
            'Static_E_4.0GHz': dvfs.make_static('E_4.0GHz')(*policy_args, metric=m_type),
            # Reactive: uses only past chunk data (causal, practical)
            'Reactive_1_Step_E': dvfs.make_reactive_1_step('E')(*policy_args, metric=m_type),
            # Oracle policies: have access to current/future timing data (not practically deployable)
            'Greedy_Oracle_E': dvfs.make_proactive_1_step('E')(*policy_args, metric=m_type),
            'MPC_Oracle_E_W5': dvfs.make_proactive_n_step('E', horizon=5)(*policy_args, metric=m_type),
            'MPC_Oracle_E_W10': dvfs.make_proactive_n_step('E', horizon=10)(*policy_args, metric=m_type),
            'Global_Oracle_E': dvfs.make_global_viterbi('E')(*policy_args, metric=m_type),
            # Industry OS governors (use proxy signal, practical)
            'Ondemand_E': dvfs.make_ondemand('E')(*policy_args, metric=m_type),
            'Conservative_E': dvfs.make_conservative('E')(*policy_args, metric=m_type),
            'Schedutil_PELT_E': dvfs.make_schedutil_pelt('E')(*policy_args, metric=m_type),
            # Academic prediction / online learning (practical)
            'EWMA_E': dvfs.make_ewma_dvfs('E')(*policy_args, metric=m_type),
            'UCB1_E': dvfs.make_ucb1_dvfs('E')(*policy_args, metric=m_type),
            'Random_E': dvfs.make_random_dvfs('E')(*policy_args, metric=m_type),
        }
        
        # 3. Heterogeneous Scheduling Policies (P vs E)
        hetero_traces = {
            # Global oracle bound (full P+E x freq Viterbi)
            'Proactive_Hetero_Oracle': sched.run_proactive_hetero_oracle(*policy_args, metric=m_type),
            # Iso-frequency oracles: ablation separating core selection from freq scaling
            'IsoFreq_Oracle_1.0GHz': sched.make_global_oracle_fixed_freq('1.0GHz')(*policy_args, metric=m_type),
            'IsoFreq_Oracle_2.0GHz': sched.make_global_oracle_fixed_freq('2.0GHz')(*policy_args, metric=m_type),
            'IsoFreq_Oracle_3.0GHz': sched.make_global_oracle_fixed_freq('3.0GHz')(*policy_args, metric=m_type),
            'IsoFreq_Oracle_4.0GHz': sched.make_global_oracle_fixed_freq('4.0GHz')(*policy_args, metric=m_type),
            # Industry: Linux EAS variants
            'EAS_Hetero': sched.make_eas_hetero()(*policy_args, metric=m_type),
            'EAS_With_DVFS': sched.make_eas_with_dvfs()(*policy_args, metric=m_type),
            # Industry: ARM big.LITTLE hysteresis migration
            'Threshold_Migration': sched.make_threshold_migration()(*policy_args, metric=m_type),
            # Industry: Intel Thread Director classification + DVFS
            'Thread_Director': sched.make_thread_director()(*policy_args, metric=m_type),
            # Simplified EAS (2-config reactive baseline)
            'Micro_EAS': sched.run_micro_eas(*policy_args, metric=m_type),
            # Online learning across full P+E space
            'UCB1_Hetero': sched.make_ucb1_hetero()(*policy_args, metric=m_type),
        }
        
        # 4. Combined DVFS + Migration Policies
        combined_traces = {
            # Reactive history-lookback (practical, causal)
            'Reactive_Combined_W1': comb.make_reactive_n_step_combined(lookback=1)(*policy_args, metric=m_type),
            'Reactive_Combined_W5': comb.make_reactive_n_step_combined(lookback=5)(*policy_args, metric=m_type),
            'Reactive_Combined_W10': comb.make_reactive_n_step_combined(lookback=10)(*policy_args, metric=m_type),
            # MPC oracle lookahead (requires future timing data)
            'MPC_Oracle_Combined_W1': comb.make_proactive_n_step_combined(horizon=1)(*policy_args, metric=m_type),
            'MPC_Oracle_Combined_W5': comb.make_proactive_n_step_combined(horizon=5)(*policy_args, metric=m_type),
            'MPC_Oracle_Combined_W10': comb.make_proactive_n_step_combined(horizon=10)(*policy_args, metric=m_type),
            # Industry hetero policies shown alongside MPC for direct comparison
            'EAS_Hetero': hetero_traces['EAS_Hetero'],
            'EAS_With_DVFS': hetero_traces['EAS_With_DVFS'],
            'Threshold_Migration': hetero_traces['Threshold_Migration'],
            'Thread_Director': hetero_traces['Thread_Director'],
            'UCB1_Hetero': hetero_traces['UCB1_Hetero'],
            # Oracle bound
            'Proactive_Hetero_Oracle': hetero_traces['Proactive_Hetero_Oracle'],
        }

        # Aggregate Results
        all_traces = {**p_traces, **e_traces, **hetero_traces, **combined_traces}
        for name, tr in all_traces.items():
            if len(tr) > 0:
                summary_results.append({'Workload': wl, 'Phase': ph, 'Metric': m_type, 'Policy': name, 'Final_Value': tr[-1]})
        
        # Generate Modular Plots
        dvfs_traces = {**p_traces, **e_traces} # Combine them so they output to the same DVFS file
        # Generate Modular Plots (Separated cleanly to prevent dict overwriting)
        plotter.generate_phase_plots(wl, ph, m_type, p_traces, min_len, bar_dir, trace_dir, "PDVFS")
        plotter.generate_phase_plots(wl, ph, m_type, e_traces, min_len, bar_dir, trace_dir, "EDVFS")
        plotter.generate_phase_plots(wl, ph, m_type, hetero_traces, min_len, bar_dir, trace_dir, "HETERO")
        plotter.generate_phase_plots(wl, ph, m_type, combined_traces, min_len, bar_dir, trace_dir, "COMBINED")
    
    return summary_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--power_mode', type=str, default='per_sample', choices=['per_sample', 'baseline'],
                         help="'per_sample' uses measured per-chunk power (default); "
                              "'baseline' uses the fixed get_power_w lookup table for all configs.")
    args = parser.parse_args()
    
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    bar_dir, trace_dir = output_path / "bar_plots_mod", output_path / "trace_plots_mod"
    bar_dir.mkdir(parents=True, exist_ok=True); trace_dir.mkdir(parents=True, exist_ok=True)
    
    pattern = re.compile(r"speedups_([PE]_[0-9.]+GHz)_(.+)_phase(\d+)\.csv")
    pairs = set(m.groups()[1:] for f in input_path.glob("speedups_*.csv") if (m := pattern.search(f.name)))
    configs = ['E_1.0GHz', 'E_2.0GHz', 'E_3.0GHz', 'E_4.0GHz', 'P_1.0GHz', 'P_2.0GHz', 'P_3.0GHz', 'P_4.0GHz', 'P_5.0GHz']
    
    print(f"Starting Modular Simulator for {len(pairs)} workloads...")
    
    all_summary = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_workload, wl, ph, pairs, input_path, bar_dir, trace_dir, configs, args.power_mode): (wl, ph) for wl, ph in pairs}
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res: all_summary.extend(res)
            
    if all_summary:
        import pandas as pd
        df = pd.DataFrame(all_summary)
        df.to_csv(output_path / "all_phases_summary.csv", index=False)
        print(f"\\nSaved aggregated summary to {output_path / 'all_phases_summary.csv'}")
    else:
        print("\\nNo valid data processed.")

if __name__ == "__main__":
    main()
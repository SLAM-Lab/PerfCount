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

# ==========================================
# GLOBAL HARDWARE DEFINES
# ==========================================
MIG_LAT_S  = 0.000100   # 100us Migration Latency
MIG_NRG_J  = 0.000015   # 15uJ Migration Energy
DVFS_LAT_S = 0.000040   # 40us Fallback Frequency Scale Latency
DVFS_NRG_J = 0.000003   # 3uJ Fallback Frequency Scale Energy

# Transition Latencies (microseconds) from physical measurements
P_LAT_US = {
    (1.0, 2.0): 105.34, (1.0, 3.0): 104.38, (1.0, 4.0): 97.80,
    (2.0, 1.0): 71.88,  (2.0, 3.0): 64.22,  (2.0, 4.0): 62.33,
    (3.0, 1.0): 70.23,  (3.0, 2.0): 44.36,  (3.0, 4.0): 40.41,
    (4.0, 1.0): 64.81,  (4.0, 2.0): 39.36,  (4.0, 3.0): 33.56
}

E_LAT_US = {
    (1.0, 2.0): 171.41, (1.0, 3.0): 177.65, (1.0, 4.0): 165.83,
    (2.0, 1.0): 156.87, (2.0, 3.0): 154.76, (2.0, 4.0): 152.71,
    (3.0, 1.0): 160.40, (3.0, 2.0): 135.03, (3.0, 4.0): 134.13,
    (4.0, 1.0): 158.26, (4.0, 2.0): 124.65, (4.0, 3.0): 101.38
}

# Transition Energy Overhead (Joules) [Peak Watts * Stall Time]
P_NRG_J = {
    (1.0, 2.0): 0.001054, (1.0, 3.0): 0.000963, (1.0, 4.0): 0.001320,
    (2.0, 1.0): 0.000615, (2.0, 3.0): 0.000538, (2.0, 4.0): 0.000643,
    (3.0, 1.0): 0.000480, (3.0, 2.0): 0.000387, (3.0, 4.0): 0.000398,
    (4.0, 1.0): 0.000601, (4.0, 2.0): 0.000581, (4.0, 3.0): 0.000296
}

E_NRG_J = {
    (1.0, 2.0): 0.004799, (1.0, 3.0): 0.006063, (1.0, 4.0): 0.011006,
    (2.0, 1.0): 0.004422, (2.0, 3.0): 0.004630, (2.0, 4.0): 0.009178,
    (3.0, 1.0): 0.005030, (3.0, 2.0): 0.003821, (3.0, 4.0): 0.008694,
    (4.0, 1.0): 0.009941, (4.0, 2.0): 0.008859, (4.0, 3.0): 0.006279
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

def process_workload(wl, ph, pairs, input_path, bar_dir, trace_dir, configs):
    data = load_phase_data(wl, ph, input_path, configs)
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
            'Static_P_1.0GHz': dvfs.make_static('P_1.0GHz')(*policy_args, metric=m_type),
            'Static_P_2.0GHz': dvfs.make_static('P_2.0GHz')(*policy_args, metric=m_type),
            'Static_P_3.0GHz': dvfs.make_static('P_3.0GHz')(*policy_args, metric=m_type),
            'Static_P_4.0GHz': dvfs.make_static('P_4.0GHz')(*policy_args, metric=m_type),
            'Reactive_1_Step_P': dvfs.make_reactive_1_step('P')(*policy_args, metric=m_type),
            'Proactive_1_Step_P': dvfs.make_proactive_1_step('P')(*policy_args, metric=m_type),
            'Proactive_N_Step_P_Window_5': dvfs.make_proactive_n_step('P', horizon=5)(*policy_args, metric=m_type),
            'Proactive_N_Step_P_Window_10': dvfs.make_proactive_n_step('P', horizon=10)(*policy_args, metric=m_type),
            'Proactive_P_Oracle': dvfs.make_global_viterbi('P')(*policy_args, metric=m_type),
#            'Linux_Schedutil_Proxy': dvfs.run_linux_schedutil(*policy_args, metric=m_type),
#            'Intel_HWP_Proxy': dvfs.run_intel_hwp(*policy_args, metric=m_type)
        }
        
        # 2. E-Core DVFS Policies
        e_traces = {
            'Static_E_1.0GHz': dvfs.make_static('E_1.0GHz')(*policy_args, metric=m_type),
            'Static_E_2.0GHz': dvfs.make_static('E_2.0GHz')(*policy_args, metric=m_type),
            'Static_E_3.0GHz': dvfs.make_static('E_3.0GHz')(*policy_args, metric=m_type),
            'Static_E_4.0GHz': dvfs.make_static('E_4.0GHz')(*policy_args, metric=m_type),
            'Reactive_1_Step_E': dvfs.make_reactive_1_step('E')(*policy_args, metric=m_type),
            'Proactive_1_Step_E': dvfs.make_proactive_1_step('E')(*policy_args, metric=m_type),
            'Proactive_N_Step_E_Window_5': dvfs.make_proactive_n_step('E', horizon=5)(*policy_args, metric=m_type),
            'Proactive_N_Step_E_Window_10': dvfs.make_proactive_n_step('E', horizon=10)(*policy_args, metric=m_type),
            'Proactive_E_Oracle': dvfs.make_global_viterbi('E')(*policy_args, metric=m_type)
        }
        
        # 3. Heterogeneous Scheduling Policies (P vs E)
        hetero_traces = {
            'Proactive_Hetero_Oracle': sched.run_proactive_hetero_oracle(*policy_args, metric=m_type)
        }
        
        # 4. Combined DVFS + Migration Policies
        combined_traces = {
            'Reactive_N_Step_Combined_Window_1': comb.make_reactive_n_step_combined(lookback=1)(*policy_args, metric=m_type),
            'Reactive_N_Step_Combined_Window_5': comb.make_reactive_n_step_combined(lookback=5)(*policy_args, metric=m_type),
            'Reactive_N_Step_Combined_Window_10': comb.make_reactive_n_step_combined(lookback=10)(*policy_args, metric=m_type),
            'Proactive_N_Step_Combined_Window_1': comb.make_proactive_n_step_combined(horizon=1)(*policy_args, metric=m_type),
            'Proactive_N_Step_Combined_Window_5': comb.make_proactive_n_step_combined(horizon=5)(*policy_args, metric=m_type),
            'Proactive_N_Step_Combined_Window_10': comb.make_proactive_n_step_combined(horizon=10)(*policy_args, metric=m_type),
            'Proactive_Hetero_Oracle': hetero_traces['Proactive_Hetero_Oracle'] # Compare against absolute optimal
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
        futures = {executor.submit(process_workload, wl, ph, pairs, input_path, bar_dir, trace_dir, configs): (wl, ph) for wl, ph in pairs}
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
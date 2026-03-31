# main.py
import argparse
import re
import numpy as np
from pathlib import Path
import concurrent.futures

# Import our custom modules
from scripts.evaluation.scheduling.src.data_loader import load_phase_data
import scripts.evaluation.scheduling.src.dvfs_policies as dvfs
import scripts.evaluation.scheduling.src.scheduling_policies as sched
import scripts.evaluation.scheduling.src.plotter as plotter
import scripts.evaluation.scheduling.src.combined_policies as comb

# ==========================================
# GLOBAL HARDWARE DEFINES
# ==========================================
MIG_LAT_S  = 0.000100   # 100us Migration Latency
MIG_NRG_J  = 0.000015   # 15uJ Migration Energy
DVFS_LAT_S = 0.000040   # 40us Frequency Scale Latency
DVFS_NRG_J = 0.000003   # 3uJ Frequency Scale Energy

def build_transition_matrices(configs):
    n = len(configs)
    lat_mat, nrg_mat = np.zeros((n, n)), np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j: continue
            if configs[i][0] != configs[j][0]:
                lat_mat[i, j], nrg_mat[i, j] = MIG_LAT_S, MIG_NRG_J
            else:
                lat_mat[i, j], nrg_mat[i, j] = DVFS_LAT_S, DVFS_NRG_J
    return lat_mat, nrg_mat

def process_workload(wl, ph, input_path, bar_dir, trace_dir, configs):
    # 1. Load Data
    data = load_phase_data(wl, ph, input_path, configs)
    if not data: return None
    time_mat, energy_mat, proxy_signal, valid_configs, min_len = data
    
    trans_lat, trans_nrg = build_transition_matrices(configs)
    summary_results = []

    # 2. Define Policy Registries
    dvfs_registry = {
        # OS / Hardware Baselines
        'Linux_Schedutil_DVFS': dvfs.run_linux_schedutil,
        'Intel_HWP_DVFS': dvfs.run_intel_hwp,

        'Static_E_1.0GHz': dvfs.make_static('E_1.0GHz'),
        'Static_E_2.0GHz': dvfs.make_static('E_2.0GHz'),
        'Static_E_3.0GHz': dvfs.make_static('E_3.0GHz'),
        'Static_E_4.0GHz': dvfs.make_static('E_4.0GHz'),
        
        'Static_P_1.0GHz': dvfs.make_static('P_1.0GHz'),
        'Static_P_2.0GHz': dvfs.make_static('P_2.0GHz'),
        'Static_P_3.0GHz': dvfs.make_static('P_3.0GHz'),
        'Static_P_4.0GHz': dvfs.make_static('P_4.0GHz'),
        
        # Reactive Policies
        'Reactive_1_Step_P': dvfs.make_reactive_1_step('P'),
        'Reactive_1_Step_E': dvfs.make_reactive_1_step('E'),
        
        # Proactive MPC Sweeps (P-Core)
        'Proactive_1_Step_P': dvfs.make_proactive_n_step('P', 1),
        'Proactive_2_Step_P': dvfs.make_proactive_n_step('P', 2),
        'Proactive_4_Step_P': dvfs.make_proactive_n_step('P', 4),
        'Proactive_8_Step_P': dvfs.make_proactive_n_step('P', 8),
        'Proactive_16_Step_P': dvfs.make_proactive_n_step('P', 16),
        'Proactive_32_Step_P': dvfs.make_proactive_n_step('P', 32),
        
        # Proactive MPC Sweeps (E-Core)
        'Proactive_1_Step_E': dvfs.make_proactive_n_step('E', 1),
        'Proactive_2_Step_E': dvfs.make_proactive_n_step('E', 2),
        'Proactive_4_Step_E': dvfs.make_proactive_n_step('E', 4),
        'Proactive_8_Step_E': dvfs.make_proactive_n_step('E', 8),
        'Proactive_16_Step_E': dvfs.make_proactive_n_step('E', 16),
        'Proactive_32_Step_E': dvfs.make_proactive_n_step('E', 32),
        
        # Global Oracles
        'Global_Oracle_P': dvfs.make_global_viterbi('P'),
        'Global_Oracle_E': dvfs.make_global_viterbi('E'),
    }

    combined_registry = {
        # --- Reactive Combined Sweeps ---
        'Reactive_1_Step_Combined': comb.make_reactive_n_step_combined(1),
        'Reactive_2_Step_Combined': comb.make_reactive_n_step_combined(2),
        'Reactive_4_Step_Combined': comb.make_reactive_n_step_combined(4),
        'Reactive_8_Step_Combined': comb.make_reactive_n_step_combined(8),
        'Reactive_16_Step_Combined': comb.make_reactive_n_step_combined(16),
        'Reactive_32_Step_Combined': comb.make_reactive_n_step_combined(32),

        # --- Proactive Combined Sweeps ---
        'Proactive_1_Step_Combined': comb.make_proactive_n_step_combined(1),
        'Proactive_2_Step_Combined': comb.make_proactive_n_step_combined(2),
        'Proactive_4_Step_Combined': comb.make_proactive_n_step_combined(4),
        'Proactive_8_Step_Combined': comb.make_proactive_n_step_combined(8),
        'Proactive_16_Step_Combined': comb.make_proactive_n_step_combined(16),
        'Proactive_32_Step_Combined': comb.make_proactive_n_step_combined(32),

        # --- Global Oracle ---
        'Global_Oracle_Combined': comb.run_global_oracle_combined,
    }

    # 2. Define Heterogeneous Policy Registry (Frequency Sweep)
    target_freqs = ['1.0GHz', '2.0GHz', '3.0GHz', '4.0GHz']
    hetero_registry = {
        # Keep the cross-frequency OS baseline for comparison
        'Proactive_Hetero_Oracle': sched.run_proactive_hetero_oracle,
        'Micro_EAS_Hetero': sched.run_micro_eas 
    }

    for f in target_freqs:
        # --- Static Baselines per frequency ---
        hetero_registry[f'Static_P_{f}'] = dvfs.make_static(f'P_{f}')
        hetero_registry[f'Static_E_{f}'] = dvfs.make_static(f'E_{f}')

        # --- Reactive History Lookback Sweeps ---
        for n in [1, 2, 4, 8, 16, 32]:
            name = f'Reactive_{n}_Step_Hetero_{f}'
            hetero_registry[name] = sched.make_reactive_n_step_fixed_freq(n, f)

        # --- Proactive Lookahead Sweeps ---
        for n in [1, 2, 4, 8, 16, 32]:
            name = f'Proactive_{n}_Step_Hetero_{f}'
            hetero_registry[name] = sched.make_proactive_n_step_fixed_freq(n, f)

        # --- Global Oracle per frequency ---
        hetero_registry[f'Global_Oracle_Hetero_{f}'] = sched.make_global_oracle_fixed_freq(f)

    for m_type in ['EDP', 'ED2P']:
        all_traces = {}
        
        # Execute DVFS Policies
        for name, func in dvfs_registry.items():
            all_traces[name] = func(time_mat, energy_mat, proxy_signal, configs, valid_configs, trans_lat, trans_nrg, m_type)
            
        # Execute Hetero Policies
        for name, func in hetero_registry.items():
            all_traces[name] = func(time_mat, energy_mat, proxy_signal, configs, valid_configs, trans_lat, trans_nrg, m_type)
            

        # Execute Combined Policies
        for name, func in combined_registry.items():
            all_traces[name] = func(time_mat, energy_mat, proxy_signal, configs, valid_configs, trans_lat, trans_nrg, m_type)
        # Extract finals and Plot
        finals = {k: v[-1] for k, v in all_traces.items()}
        res_dict = {'Workload': wl, 'Phase': ph, 'Metric': m_type}
        res_dict.update(finals)
        summary_results.append(res_dict)
        
        dvfs_traces = {k: all_traces[k] for k in dvfs_registry.keys() | {'Proactive_Hetero_Oracle'}}
        hetero_traces = {k: all_traces[k] for k in hetero_registry.keys()}
        combined_traces = {k: all_traces[k] for k in combined_registry.keys()}
        
        plotter.generate_phase_plots(wl, ph, m_type, dvfs_traces, min_len, bar_dir, trace_dir, "DVFS")
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
    
    master_results = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_workload, wl, ph, input_path, bar_dir, trace_dir, configs): (wl, ph) for wl, ph in pairs}
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res: master_results.extend(res)
            
    plotter.save_master_csv(master_results, output_path)

if __name__ == "__main__":
    main()
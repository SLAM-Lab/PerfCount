# data_loader.py
import pandas as pd
import numpy as np

def get_power_w(config_str):
    power_map = {
        'P_1.0GHz': 0.58, 'P_2.0GHz': 2.14, 'P_3.0GHz': 5.50, 'P_4.0GHz': 13.17, 'P_5.0GHz': 29.77,
        'E_1.0GHz': 0.44, 'E_2.0GHz': 1.38, 'E_3.0GHz': 4.70, 'E_4.0GHz': 16.89
    }
    return power_map.get(config_str, 10.0)

def load_phase_data(wl, ph, input_path, configs):
    speedup_files = list(input_path.glob(f"speedups_*_{wl}_phase{ph}.csv"))
    if not speedup_files: return None
    
    time_dict = {}
    power_dict = {}
    for sf in speedup_files:
        try:
            df = pd.read_csv(sf).dropna()
            base_cols = [c for c in df.columns if c.startswith('Time_')]
            if not base_cols: continue

            base_cfg = base_cols[0].replace('Time_', '')
            if base_cfg not in time_dict:
                time_dict[base_cfg] = df[base_cols[0]].values

            for col in df.columns:
                if col.startswith('Speedup_'):
                    target_cfg = col.split('_vs_')[0].replace('Speedup_', '')
                    spds = np.where(df[col].values == 0, 1e-9, df[col].values)
                    time_dict[target_cfg] = df[base_cols[0]].values / spds
                elif col.startswith('Power_'):
                    power_cfg = col.replace('Power_', '')
                    if power_cfg not in power_dict:
                        power_dict[power_cfg] = df[col].values
        except Exception: continue

    if not time_dict: return None

    min_len = min(len(arr) for arr in time_dict.values())
    valid_configs = [c for c in configs if c in time_dict]
    if not valid_configs: return None

    time_mat = np.full((min_len, len(configs)), 1e6)
    for i, cfg in enumerate(configs):
        if cfg in valid_configs:
            time_mat[:, i] = time_dict[cfg][:min_len]

    # Use measured per-chunk power where available; fall back to the fixed
    # get_power_w lookup table for configs with no measured power data
    # (e.g. the P_5.0GHz placeholder).
    power_mat = np.zeros((min_len, len(configs)))
    for i, cfg in enumerate(configs):
        if cfg in power_dict:
            power_mat[:, i] = power_dict[cfg][:min_len]
        else:
            power_mat[:, i] = get_power_w(cfg)

    energy_mat = time_mat * power_mat

    t_slow = time_mat[:, configs.index('E_1.0GHz')] if 'E_1.0GHz' in valid_configs else time_mat[:, configs.index(valid_configs[0])]
    t_fast = time_mat[:, configs.index('P_5.0GHz')] if 'P_5.0GHz' in valid_configs else time_mat[:, configs.index(valid_configs[-1])]
    proxy_signal = t_slow / (t_fast + 1e-9)
    
    return time_mat, energy_mat, proxy_signal, valid_configs, min_len
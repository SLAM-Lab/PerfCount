# data_loader.py
import pandas as pd
import numpy as np

def get_power_w(config_str):
    power_map = {
        'P_1.0GHz': 0.58, 'P_2.0GHz': 2.14, 'P_3.0GHz': 5.50, 'P_4.0GHz': 13.17, 'P_5.0GHz': 29.77,
        'E_1.0GHz': 0.44, 'E_2.0GHz': 1.38, 'E_3.0GHz': 4.70, 'E_4.0GHz': 16.89
    }
    return power_map.get(config_str, 10.0)

P_MODEL_FREQS = [1.0, 2.0, 3.0, 4.0]


def _load_speedup_dict(speedup_files):
    """Parse speedup CSV files into (time_dict, power_dict) keyed by config string."""
    time_dict = {}
    power_dict = {}
    for sf in speedup_files:
        try:
            df = pd.read_csv(sf).dropna()
            base_cols = [c for c in df.columns if c.startswith('Time_')]
            if not base_cols:
                continue
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
        except Exception:
            continue
    return time_dict, power_dict


def load_phase_data(wl, ph, input_path, configs, power_mode='per_sample',
                    model_pred_dir=None):
    speedup_files = list(input_path.glob(f"speedups_*_{wl}_phase{ph}.csv"))
    if not speedup_files:
        return None

    time_dict, power_dict = _load_speedup_dict(speedup_files)
    if not time_dict:
        return None

    min_len = min(len(arr) for arr in time_dict.values())
    valid_configs = [c for c in configs if c in time_dict]
    if not valid_configs:
        return None

    time_mat = np.full((min_len, len(configs)), 1e6)
    for i, cfg in enumerate(configs):
        if cfg in valid_configs:
            time_mat[:, i] = time_dict[cfg][:min_len]

    # Use measured per-chunk power where available; fall back to the fixed
    # get_power_w lookup table for configs with no measured power data
    # (e.g. the P_5.0GHz placeholder). In 'baseline' mode, ignore measured
    # power entirely and use the fixed lookup table for every config.
    power_mat = np.zeros((min_len, len(configs)))
    for i, cfg in enumerate(configs):
        if power_mode == 'per_sample' and cfg in power_dict:
            power_mat[:, i] = power_dict[cfg][:min_len]
        else:
            power_mat[:, i] = get_power_w(cfg)

    energy_mat = time_mat * power_mat

    t_slow = time_mat[:, configs.index('E_1.0GHz')] if 'E_1.0GHz' in valid_configs else time_mat[:, configs.index(valid_configs[0])]
    t_fast = time_mat[:, configs.index('P_5.0GHz')] if 'P_5.0GHz' in valid_configs else time_mat[:, configs.index(valid_configs[-1])]
    proxy_signal = t_slow / (t_fast + 1e-9)

    model_time_mat = None
    if model_pred_dir is not None:
        model_time_mat = _load_model_time_mat(
            wl, ph, model_pred_dir, configs, min_len, time_mat, power_mat
        )

    return time_mat, energy_mat, proxy_signal, valid_configs, min_len, model_time_mat


def _load_model_time_mat(wl, ph, model_pred_dir, configs, min_len, oracle_time_mat, power_mat):
    """Load model-predicted speedups for all 4 P-core source frequencies.

    Returns model_time_mat of shape (4, min_len, len(configs)):
      axis 0: source P-core frequency index (0=1.0GHz … 3=4.0GHz)
      axis 1: chunk index
      axis 2: config index (matches configs list)

    Diagonal entries (src_cfg == tgt_cfg in the P-core sense) use oracle time.
    Off-diagonal P-core entries use model-predicted times.
    Non-P-core and P_5.0GHz entries remain at 1e6 (huge, scheduler ignores them).
    """
    from pathlib import Path
    n_src = len(P_MODEL_FREQS)
    model_time_mat = np.full((n_src, min_len, len(configs)), 1e6)

    for si, src_freq in enumerate(P_MODEL_FREQS):
        src_ghz = f"{src_freq:.1f}GHz"
        src_cfg = f"P_{src_ghz}"
        pred_dir = Path(model_pred_dir) / f"speedups_from_P_{src_ghz}"
        pred_file = pred_dir / f"speedups_P_{src_ghz}_{wl}_phase{ph}.csv"

        if not pred_file.exists():
            # Fall back to oracle times for this source freq slice
            for ci, cfg in enumerate(configs):
                model_time_mat[si, :, ci] = oracle_time_mat[:min_len, ci]
            continue

        try:
            df = pd.read_csv(pred_file).dropna()
        except Exception:
            for ci, cfg in enumerate(configs):
                model_time_mat[si, :, ci] = oracle_time_mat[:min_len, ci]
            continue

        # Ground-truth time at source config
        time_src_col = f"Time_P_{src_ghz}"
        if time_src_col not in df.columns:
            continue
        time_src = df[time_src_col].values

        n = min(min_len, len(time_src))

        # Src config itself: oracle time (diagonal)
        if src_cfg in configs:
            ci = configs.index(src_cfg)
            model_time_mat[si, :n, ci] = time_src[:n]

        # Target P-core configs: predicted time = time_src / speedup
        for col in df.columns:
            if col.startswith("Speedup_P_") and "_vs_P_" in col:
                tgt_cfg = col.split("_vs_")[0].replace("Speedup_", "")
                if tgt_cfg not in configs:
                    continue
                ci = configs.index(tgt_cfg)
                spds = np.where(df[col].values[:n] == 0, 1e-9, df[col].values[:n])
                model_time_mat[si, :n, ci] = time_src[:n] / spds

    return model_time_mat
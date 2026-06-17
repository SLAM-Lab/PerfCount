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
E_MODEL_FREQS = [1.0, 2.0, 3.0, 4.0]

# Source-config ordering for the cross-proc time tensor (axis 0).
# E-cores first (indices 0-3), then P-cores (indices 4-7).
ALL_MODEL_CONFIGS = [
    'E_1.0GHz', 'E_2.0GHz', 'E_3.0GHz', 'E_4.0GHz',
    'P_1.0GHz', 'P_2.0GHz', 'P_3.0GHz', 'P_4.0GHz',
]


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
                    model_pred_dir=None, e_model_pred_dir=None, cross_proc_pred_dir=None):
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

    e_model_time_mat = None
    if e_model_pred_dir is not None:
        e_model_time_mat = _load_e_model_time_mat(
            wl, ph, e_model_pred_dir, configs, min_len, time_mat, power_mat
        )

    cross_proc_time_mat = None
    if cross_proc_pred_dir is not None:
        cross_proc_time_mat = _load_cross_proc_time_mat(
            wl, ph, cross_proc_pred_dir, configs, min_len, time_mat
        )

    full_model_time_mat = None
    if model_pred_dir is not None and cross_proc_pred_dir is not None:
        full_model_time_mat = _load_full_model_time_mat(
            model_time_mat, cross_proc_time_mat
        )

    return (time_mat, energy_mat, proxy_signal, valid_configs, min_len,
            model_time_mat, e_model_time_mat, cross_proc_time_mat, full_model_time_mat)


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


def _load_e_model_time_mat(wl, ph, e_model_pred_dir, configs, min_len, oracle_time_mat, power_mat):
    """Load model-predicted speedups for all 4 E-core source frequencies.

    Mirrors _load_model_time_mat but for E-core (cpu16) cross-freq predictions.
    Returns e_model_time_mat of shape (4, min_len, len(configs)):
      axis 0: source E-core frequency index (0=1.0GHz … 3=4.0GHz)
      axis 1: chunk index
      axis 2: config index (matches configs list)

    Expects CSVs at: e_model_pred_dir/speedups_from_E_{src_ghz}/speedups_E_{src_ghz}_{wl}_phase{ph}.csv
    Columns: Time_E_{src_ghz}, Speedup_E_{tgt_ghz}_vs_E_{src_ghz}
    """
    from pathlib import Path
    n_src = len(E_MODEL_FREQS)
    e_model_time_mat = np.full((n_src, min_len, len(configs)), 1e6)

    for si, src_freq in enumerate(E_MODEL_FREQS):
        src_ghz = f"{src_freq:.1f}GHz"
        src_cfg = f"E_{src_ghz}"
        pred_dir = Path(e_model_pred_dir) / f"speedups_from_E_{src_ghz}"
        pred_file = pred_dir / f"speedups_E_{src_ghz}_{wl}_phase{ph}.csv"

        if not pred_file.exists():
            for ci, cfg in enumerate(configs):
                e_model_time_mat[si, :, ci] = oracle_time_mat[:min_len, ci]
            continue

        try:
            df = pd.read_csv(pred_file).dropna()
        except Exception:
            for ci, cfg in enumerate(configs):
                e_model_time_mat[si, :, ci] = oracle_time_mat[:min_len, ci]
            continue

        time_src_col = f"Time_E_{src_ghz}"
        if time_src_col not in df.columns:
            continue
        time_src = df[time_src_col].values
        n = min(min_len, len(time_src))

        if src_cfg in configs:
            ci = configs.index(src_cfg)
            e_model_time_mat[si, :n, ci] = time_src[:n]

        for col in df.columns:
            if col.startswith("Speedup_E_") and "_vs_E_" in col:
                tgt_cfg = col.split("_vs_")[0].replace("Speedup_", "")
                if tgt_cfg not in configs:
                    continue
                ci = configs.index(tgt_cfg)
                spds = np.where(df[col].values[:n] == 0, 1e-9, df[col].values[:n])
                e_model_time_mat[si, :n, ci] = time_src[:n] / spds

    return e_model_time_mat



def _load_cross_proc_time_mat(wl, ph, cross_proc_pred_dir, configs, min_len, oracle_time_mat):
    """Load cross-proc model predictions for all 8 source configs (E_1-4, P_1-4).

    Returns cross_proc_time_mat of shape (8, min_len, len(configs)):
      axis 0: source config index per ALL_MODEL_CONFIGS (0=E_1 … 7=P_4)
      axis 1: chunk index
      axis 2: config index (matches configs list)

    Diagonal entries (src_cfg == tgt_cfg) use oracle time.
    Cross-proc entries use model-predicted times (P↔E only).
    Same-core different-freq entries remain at 1e6 (scheduler avoids them).
    """
    from pathlib import Path
    n_src = len(ALL_MODEL_CONFIGS)
    cross_proc_mat = np.full((n_src, min_len, len(configs)), 1e6)

    for si, src_cfg in enumerate(ALL_MODEL_CONFIGS):
        src_core, src_freq_str = src_cfg.split('_', 1)
        tgt_core = 'E' if src_core == 'P' else 'P'
        src_ghz = src_freq_str

        pred_dir = Path(cross_proc_pred_dir) / f"speedups_from_{src_core}_{src_ghz}"
        pred_file = pred_dir / f"{wl}_phase{ph}.csv"

        # Set diagonal: src → src = oracle time
        if src_cfg in configs:
            src_ci = configs.index(src_cfg)
            cross_proc_mat[si, :, src_ci] = oracle_time_mat[:min_len, src_ci]

        if not pred_file.exists():
            continue

        try:
            df = pd.read_csv(pred_file).dropna()
        except Exception:
            continue

        time_col = f"Time_{src_core}_{src_ghz}"
        if time_col not in df.columns:
            continue
        time_src = df[time_col].values
        n = min(min_len, len(time_src))

        # Cross-proc entries: predicted time = time_src / speedup
        for col in df.columns:
            if col.startswith(f"Speedup_{tgt_core}_") and f"_vs_{src_core}_" in col:
                tgt_cfg = col.split("_vs_")[0].replace("Speedup_", "")
                if tgt_cfg not in configs:
                    continue
                tgt_ci = configs.index(tgt_cfg)
                spds = np.where(df[col].values[:n] == 0, 1e-9, df[col].values[:n])
                cross_proc_mat[si, :n, tgt_ci] = time_src[:n] / spds

    return cross_proc_mat


def _load_full_model_time_mat(model_time_mat, cross_proc_time_mat):
    """Combine cross-freq (P->P) and cross-proc (P<->E) predictions.

    Returns full_model_time_mat of shape (8, n_chunks, n_configs):
      axis 0: source config index per ALL_MODEL_CONFIGS (0=E_1 ... 7=P_4)
      Rows 0-3 (E sources): E->P cross-proc predictions + E->E diagonal (oracle).
      Rows 4-7 (P sources): P->P cross-freq + P->E cross-proc (elementwise min,
        since each matrix has 1e6 where it has no prediction and real values
        where it does, so min selects the real prediction in each position).
    """
    full_mat = cross_proc_time_mat.copy()
    for si in range(4):
        full_mat[4 + si] = np.minimum(cross_proc_time_mat[4 + si], model_time_mat[si])
    return full_mat
# data_loader.py
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Prediction-fallback accounting.
#
# When a prediction CSV is missing the loaders silently substitute oracle times.
# That is a reasonable default for a partially-populated prediction set, but it
# makes a *model* policy secretly identical to its oracle counterpart, which is
# indistinguishable from a real result in the output CSV. A dump/sim race once
# produced exactly this: an entire suite's forecast policies collapsed onto
# perfect-future and looked like a genuine (excellent) result.
#
# Every fallback is now counted. main.py reports the totals at the end of a run
# and, under --strict_predictions, aborts instead of substituting.
# ---------------------------------------------------------------------------
FALLBACK_COUNTS = {}
STRICT_PREDICTIONS = False


class MissingPredictionError(RuntimeError):
    pass


def _note_fallback(kind, wl, ph, detail):
    """Record (and under strict mode, raise on) a prediction-file fallback."""
    key = f"{kind}"
    FALLBACK_COUNTS[key] = FALLBACK_COUNTS.get(key, 0) + 1
    if STRICT_PREDICTIONS:
        raise MissingPredictionError(
            f"{kind}: no prediction file for {wl} phase{ph} ({detail}). "
            f"A prediction dir was passed explicitly, so falling back to oracle "
            f"times would silently fabricate a model result. Re-run the dump, or "
            f"drop --strict_predictions to allow the fallback."
        )


def reset_fallback_counts():
    FALLBACK_COUNTS.clear()


def fallback_report():
    """Human-readable summary of prediction fallbacks, or None if there were none."""
    if not FALLBACK_COUNTS:
        return None
    lines = ["PREDICTION FALLBACKS (oracle times substituted for missing predictions):"]
    for k, v in sorted(FALLBACK_COUNTS.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {k:44s} {v:6d} phase-slices")
    lines.append("  Model/forecast policies on affected phases are NOT model results.")
    return "\n".join(lines)

# Per-configuration mean power (W), from a full-run RAPL average with the idle
# baseline subtracted. Used in 'baseline' power mode, and for any config lacking
# measured per-chunk power.
POWER_W = {
    'P_1.0GHz': 0.58, 'P_2.0GHz': 2.14, 'P_3.0GHz': 5.50, 'P_4.0GHz': 13.17,
    'E_1.0GHz': 0.44, 'E_2.0GHz': 1.38, 'E_3.0GHz': 4.70, 'E_4.0GHz': 16.89,
    'L_1.0GHz': 12.50, 'B_1.0GHz': 11.73, 'B_2.0GHz': 11.51,
}


def get_power_w(config_str):
    """Mean power for a configuration. Raises on an unknown config.

    This previously returned a default of 10.0 W for anything unrecognized, which
    is the same failure class as the silent oracle fallback: a plausible-looking
    number in place of an error. A typo'd or unmeasured config would silently get
    a fabricated power and land in the energy totals looking like a real result.
    """
    try:
        return POWER_W[config_str]
    except KeyError:
        raise KeyError(
            f"No measured power for config {config_str!r}. Known configs: "
            f"{sorted(POWER_W)}. Add its measured mean power to POWER_W rather "
            f"than allowing a default, which would silently fabricate energy."
        ) from None

P_MODEL_FREQS = [1.0, 2.0, 3.0, 4.0]
E_MODEL_FREQS = [1.0, 2.0, 3.0, 4.0]

ARM_L_MODEL_FREQS = [1.0]
ARM_B_MODEL_FREQS = [1.0]

# Source-config ordering for the cross-proc time tensor (axis 0).
# E-cores first (indices 0-3), then P-cores (indices 4-7).
ALL_MODEL_CONFIGS = [
    'E_1.0GHz', 'E_2.0GHz', 'E_3.0GHz', 'E_4.0GHz',
    'P_1.0GHz', 'P_2.0GHz', 'P_3.0GHz', 'P_4.0GHz',
]

ARM_ALL_MODEL_CONFIGS = [
    'L_1.0GHz',
    'B_1.0GHz',
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
    # In 'baseline' mode, ignore measured
    # power entirely and use the fixed lookup table for every config.
    power_mat = np.zeros((min_len, len(configs)))
    for i, cfg in enumerate(configs):
        if power_mode == 'per_sample' and cfg in power_dict:
            power_mat[:, i] = power_dict[cfg][:min_len]
        else:
            power_mat[:, i] = get_power_w(cfg)

    energy_mat = time_mat * power_mat

    # Proxy utilization signal: the per-chunk ratio of the slowest to the fastest
    # configuration, i.e. the workload's dynamic range. Endpoints are chosen from the
    # measured data rather than by position in `configs`, which previously made the
    # signal depend on list ordering and on a P_5.0GHz entry that has no traces.
    _valid_idx = [configs.index(c) for c in valid_configs]
    _mean_t = time_mat[:, _valid_idx].mean(axis=0)
    t_slow = time_mat[:, _valid_idx[int(np.argmax(_mean_t))]]
    t_fast = time_mat[:, _valid_idx[int(np.argmin(_mean_t))]]
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
            model_time_mat, cross_proc_time_mat, e_model_time_mat
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
    Non-P-core entries remain at 1e6 (huge, scheduler ignores them).
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
            _note_fallback('cross_freq_P: missing file', wl, ph, pred_file.name)
            for ci, cfg in enumerate(configs):
                model_time_mat[si, :, ci] = oracle_time_mat[:min_len, ci]
            continue

        try:
            df = pd.read_csv(pred_file).dropna()
        except Exception:
            _note_fallback('cross_freq_P: unreadable file', wl, ph, pred_file.name)
            for ci, cfg in enumerate(configs):
                model_time_mat[si, :, ci] = oracle_time_mat[:min_len, ci]
            continue

        # Ground-truth time at source config
        time_src_col = f"Time_P_{src_ghz}"
        if time_src_col not in df.columns:
            _note_fallback('cross_freq_P: no Time_ column', wl, ph, pred_file.name)
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
            _note_fallback('cross_freq_E: missing file', wl, ph, pred_file.name)
            for ci, cfg in enumerate(configs):
                e_model_time_mat[si, :, ci] = oracle_time_mat[:min_len, ci]
            continue

        try:
            df = pd.read_csv(pred_file).dropna()
        except Exception:
            _note_fallback('cross_freq_E: unreadable file', wl, ph, pred_file.name)
            for ci, cfg in enumerate(configs):
                e_model_time_mat[si, :, ci] = oracle_time_mat[:min_len, ci]
            continue

        time_src_col = f"Time_E_{src_ghz}"
        if time_src_col not in df.columns:
            _note_fallback('cross_freq_E: no Time_ column', wl, ph, pred_file.name)
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
            _note_fallback('cross_proc: missing file', wl, ph, pred_file.name)
            continue

        try:
            df = pd.read_csv(pred_file).dropna()
        except Exception:
            _note_fallback('cross_proc: unreadable file', wl, ph, pred_file.name)
            continue

        time_col = f"Time_{src_core}_{src_ghz}"
        if time_col not in df.columns:
            _note_fallback('cross_proc: no Time_ column', wl, ph, pred_file.name)
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


def load_workload_data(wl, phases_sorted, input_path, configs, **kwargs):
    """Load and concatenate all phases for one workload into a single long trace.

    phases_sorted: list of phase strings in temporal order (e.g. ['0', '1', '2']).
    Returns the same 9-tuple as load_phase_data. If any phase lacks a model matrix
    (e.g. cross-proc predictions not generated), that matrix is None for the whole
    workload. Phases that fail to load are silently skipped.
    """
    parts = [load_phase_data(wl, ph, input_path, configs, **kwargs)
             for ph in phases_sorted]
    parts = [p for p in parts if p is not None]
    if not parts:
        return None

    time_mat   = np.vstack([p[0] for p in parts])
    energy_mat = np.vstack([p[1] for p in parts])
    proxy      = np.concatenate([p[2] for p in parts])
    valid_configs = parts[0][3]
    min_len    = sum(p[4] for p in parts)

    def _concat_model(idx):
        mats = [p[idx] for p in parts]
        if any(m is None for m in mats):
            return None
        return np.concatenate(mats, axis=1)

    model_time_mat      = _concat_model(5)
    e_model_time_mat    = _concat_model(6)
    cross_proc_time_mat = _concat_model(7)
    full_model_time_mat = _concat_model(8)

    return (time_mat, energy_mat, proxy, valid_configs, min_len,
            model_time_mat, e_model_time_mat, cross_proc_time_mat, full_model_time_mat)


def _load_forecast_oracle_mat(wl, ph, fc_oracle_dir, configs, min_len, oracle_time_mat):
    """Forecast x Oracle: causal forecast of each config's TRUE time, no translation.

    Returns (n_src, min_len, n_configs), constant along axis 0. The forecast of config
    C is made from C's own true history, so it does not depend on where the workload
    actually ran -- unlike every other prediction tensor here, it has no real source
    axis. It is broadcast across sources so the existing model policies can read it
    unchanged (temporal='oracle' reads row i, which under this tensor is a causal
    forecast rather than ground truth).

    This fills the one empty cell of the temporal x prediction-source grid. Every other
    forecast policy conflates forecast error with cross-platform translation error;
    this one isolates forecast error alone, and so bounds what forecasting could
    achieve if translation were perfect. Like the reactive oracle it is not deployable
    (it needs the true history of configs the workload never ran on), and like the
    reactive oracle it is a bound rather than a proposal.

    Expects: {fc_oracle_dir}/{config}/{wl}_phase{ph}.csv with a Time_pred_{config} column.
    """
    from pathlib import Path
    per_cfg = np.array(oracle_time_mat[:min_len, :], dtype=float, copy=True)
    got_any = False
    for ci, cfg in enumerate(configs):
        f = Path(fc_oracle_dir) / cfg / f"{wl}_phase{ph}.csv"
        if not f.exists():
            _note_fallback('forecast_oracle: missing file', wl, ph, f"{cfg}/{f.name}")
            continue
        try:
            df = pd.read_csv(f)
        except Exception:
            _note_fallback('forecast_oracle: unreadable file', wl, ph, f.name)
            continue
        col = f'Time_pred_{cfg}'
        if col not in df.columns:
            _note_fallback('forecast_oracle: no Time_pred_ column', wl, ph, f.name)
            continue
        v = df[col].values
        n = min(min_len, len(v))
        per_cfg[:n, ci] = v[:n]
        got_any = True
    if not got_any:
        return None
    return np.broadcast_to(per_cfg, (len(ALL_MODEL_CONFIGS), min_len, len(configs))).copy()


def build_oracle_axis_mat(time_mat, configs, min_len, axis):
    """Ground-truth tensor shaped like a prediction tensor, for one prediction axis.

    Lets the simulator run the 2x2 of {cross-freq oracle, cross-freq model} x
    {cross-proc oracle, cross-proc model}. Substituting truth on one axis and the
    model on the other attributes a deficit to a specific model, rather than to
    "model error" in the aggregate. This matters because the two models are not
    equally good: the cross-frequency model tracks frequency accurately while the
    cross-processor model spans a microarchitecture gap.

    axis='cross_freq' fills same-core targets with true times (the shape produced by
    _load_model_time_mat / _load_e_model_time_mat combined); axis='cross_proc' fills
    opposite-core targets (the shape of _load_cross_proc_time_mat). Both fill the
    src->src diagonal, and leave everything else at 1e6 so the elementwise-min merge
    in _load_full_model_time_mat picks the other axis there.
    """
    n_src = len(ALL_MODEL_CONFIGS)
    out = np.full((n_src, min_len, len(configs)), 1e6)
    for si, src_cfg in enumerate(ALL_MODEL_CONFIGS):
        src_core = src_cfg.split('_', 1)[0]
        if src_cfg in configs:
            ci = configs.index(src_cfg)
            out[si, :, ci] = time_mat[:min_len, ci]
        for ci, tgt_cfg in enumerate(configs):
            tgt_core = tgt_cfg.split('_', 1)[0]
            same_core = (tgt_core == src_core)
            want = same_core if axis == 'cross_freq' else (not same_core)
            if want and tgt_cfg != src_cfg:
                out[si, :, ci] = time_mat[:min_len, ci]
    return out


def _load_full_model_time_mat(model_time_mat, cross_proc_time_mat, e_model_time_mat=None):
    """Combine cross-freq (same-core) and cross-proc (P<->E) predictions.

    Returns full_model_time_mat of shape (8, n_chunks, n_configs):
      axis 0: source config index per ALL_MODEL_CONFIGS (0=E_1 ... 7=P_4)
      Rows 0-3 (E sources): E->E cross-freq + E->P cross-proc.
      Rows 4-7 (P sources): P->P cross-freq + P->E cross-proc.

    The merge is an elementwise min: each input matrix holds 1e6 where it has no
    prediction and a real value where it does, so min selects the real prediction
    in each position. Where both hold a real value (the src->src diagonal) they
    agree, since both write the measured source time there.

    e_model_time_mat is the E-core cross-frequency tensor. Omitting it leaves every
    E_x -> E_y (y != x) entry at 1e6, which makes E-core frequency changes invisible
    to the scheduler: once the policy migrates to the E-core it can only hold that
    exact frequency or migrate back to a P config. That silently removes a quarter
    of the action space, so it is passed explicitly and its absence is worth a warning.
    """
    full_mat = cross_proc_time_mat.copy()
    for si in range(4):
        full_mat[4 + si] = np.minimum(cross_proc_time_mat[4 + si], model_time_mat[si])
    if e_model_time_mat is not None:
        for si in range(4):
            full_mat[si] = np.minimum(cross_proc_time_mat[si], e_model_time_mat[si])
    else:
        _note_fallback('full_model: no E cross-freq tensor (E-core DVFS unavailable)',
                       '-', '-', 'e_model_time_mat=None')
    return full_mat
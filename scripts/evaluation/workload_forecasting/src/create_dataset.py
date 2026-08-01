import os
import re
import glob
import sys
import numpy as np
import pandas as pd
import src.TimeSeries as ts

# Canonical ref_cycles scheduling-stall repair, shared with the cross-platform
# training and the scheduling trace generator so every consumer cleans the same way.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..', 'cross_platform_prediction'))
from shared_features import repair_ref_cycles_df

try:
    from catboost import CatBoostRegressor as _CatBoost
    _CATBOOST_AVAILABLE = True
except ImportError:
    _CATBOOST_AVAILABLE = False

# Allowed donor-suite subdirectory names for translation (cross-freq and cross-proc).
# dacapo_c2 is excluded because it lacks features required by the CatBoost models.
_ALLOWED_DONOR_SUITES = ('dacapo_c1', 'spec_2017', 'spec_2026')

BENCHMARK_NAME_RE = re.compile(
    r"^aligned_(?P<rest>.+)_(?P<freq>[\d.]+)GHz_cpu(?P<cpu>\d+)_phase(?P<phase>\d+)$"
)
BENCHMARK_NAME_NOPHASE_RE = re.compile(
    r"^aligned_(?P<rest>.+)_(?P<freq>[\d.]+)GHz_cpu(?P<cpu>\d+)$"
)

def get_raw_data(args):
    """
    Loads the pre-aligned and pre-processed CSV directly from the processed_data folder.
    Bypasses legacy traces.py and hardcoded frequency hacks.
    """
    # 1. Dynamically find the PerfCount root directory (4 folders up from src/create_dataset.py)
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

    # 2. Find the benchmark CSV recursively under the processed data directory
    data_dir = os.path.join(root_dir, 'processed_data_10M', args.dataset)
    matches = glob.glob(os.path.join(data_dir, "**", f"{args.benchmark}.csv"), recursive=True)
    if matches:
        df = pd.read_csv(matches[0])
    else:
        # No exact match — concatenate all phases (e.g. _phase0, _phase1, ...).
        # Deduplicate by phase number: sort paths alphabetically within each phase so
        # dacapo_c1 < dacapo_c2 — then keep only the first (earliest) path per phase.
        all_phase_matches = glob.glob(
            os.path.join(data_dir, "**", f"{args.benchmark}_phase*.csv"), recursive=True
        )
        phase_by_num = {}
        for p in sorted(all_phase_matches):
            num = int(re.search(r'_phase(\d+)\.csv$', p).group(1))
            if num not in phase_by_num:
                phase_by_num[num] = p
        phase_matches = [phase_by_num[n] for n in sorted(phase_by_num)]
        if not phase_matches:
            print(f"[ERROR] Could not find {args.benchmark}.csv (or phased variants) under: {data_dir}")
            return None
        phase_dfs = [pd.read_csv(f) for f in phase_matches]
        common_cols = list(set.intersection(*[set(d.columns) for d in phase_dfs]))
        df = pd.concat([d[common_cols] for d in phase_dfs], ignore_index=True)

    # Repair ref_cycles scheduling-stall artifacts before selecting counters, while
    # cpu_cycles is still present, so the forecast target is not learned from noise.
    repair_ref_cycles_df(df)

    # 4. Filter for valid requested counters
    valid_counters = [c for c in args.input_counters if c in df.columns]
    missing = [c for c in args.input_counters if c not in df.columns]
    if missing:
        print(f"[WARNING] {args.benchmark} is missing columns: {missing}")

    data_local = df[valid_counters]

    # 5. Apply cropping if specified via arguments
    if args.end_drop_count != 0:
        data_local = data_local.iloc[args.start_drop_count:-args.end_drop_count, :]
    else:
        data_local = data_local.iloc[args.start_drop_count:, :]

    return data_local

def _get_data_dir(args):
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
    return os.path.join(root_dir, 'processed_data_10M', args.dataset)


def _crop(df, args):
    if args.end_drop_count != 0:
        return df.iloc[args.start_drop_count:-args.end_drop_count, :]
    return df.iloc[args.start_drop_count:, :]


def _find_donor_files(args, columns):
    """
    Find donor CSV files for heterogeneous history injection, based on
    args.heterogeneous_mode:
      - cross_freq: same core, a different frequency
      - cross_proc: a different core, any frequency (including the same)
      - cross_proc_freq: a different core AND a different frequency

    Supports both per-phase benchmarks (aligned_..._phaseN) and concatenated
    benchmarks (aligned_... without phase suffix). In the concatenated case, all
    phase files for matching donor configs are returned; apply_heterogeneous_history
    wraps the donor index with modulo so every training row gets a replacement.

    Only donors whose header contains every column in `columns` are kept.
    """
    m = BENCHMARK_NAME_RE.match(args.benchmark)
    if m:
        rest, cur_freq, cur_cpu = m.group('rest'), m.group('freq'), m.group('cpu')
        phase_glob = f"_phase{m.group('phase')}"
    else:
        m = BENCHMARK_NAME_NOPHASE_RE.match(args.benchmark)
        if not m:
            return []
        rest, cur_freq, cur_cpu = m.group('rest'), m.group('freq'), m.group('cpu')
        phase_glob = "_phase*"

    data_dir = _get_data_dir(args)

    pattern = os.path.join(data_dir, "**", f"aligned_{rest}_*GHz_cpu*{phase_glob}.csv")
    donors = []
    for fpath in glob.glob(pattern, recursive=True):
        name = os.path.basename(fpath).replace('.csv', '')
        dm = BENCHMARK_NAME_RE.match(name)
        if not dm:
            continue
        d_freq, d_cpu = dm.group('freq'), dm.group('cpu')

        if args.heterogeneous_mode == 'cross_freq':
            if d_cpu != cur_cpu or d_freq == cur_freq:
                continue
        elif args.heterogeneous_mode == 'cross_proc':
            if d_cpu == cur_cpu:
                continue
        else:  # cross_proc_freq: other core AND other frequency
            if d_cpu == cur_cpu or d_freq == cur_freq:
                continue

        try:
            header = pd.read_csv(fpath, nrows=0).columns
        except Exception:
            continue
        if not all(c in header for c in columns):
            continue

        donors.append(fpath)

    return sorted(donors)


def _own_freq(args):
    m = BENCHMARK_NAME_RE.match(args.benchmark) or BENCHMARK_NAME_NOPHASE_RE.match(args.benchmark)
    return float(m.group('freq')) if m else 0.0


def _own_cpu(args):
    m = BENCHMARK_NAME_RE.match(args.benchmark) or BENCHMARK_NAME_NOPHASE_RE.match(args.benchmark)
    return m.group('cpu') if m else None


def add_heterogeneity_columns(args, data):
    """
    Append het_flag (0/1) and het_source_freq (GHz) columns, defaulting to
    "not heterogeneous, sourced from this trace's own frequency".
    """
    data = data.copy()
    data['het_flag'] = 0
    data['het_source_freq'] = _own_freq(args)
    return data


_SPEC_NUM_RE = re.compile(r'^spec_(\d+)\.')


def _bench_suite_dir_cross_freq(bench_name):
    """Suite subdirectory in the cross-frequency model tree.

    dacapo uses the unpruned dacapo_c1 tree. The pruned variant covers only 17 of
    the 22 dacapo benchmarks (it lacks cassandra, h2o, kafka, tradebeans, tradesoap),
    so pointing here at dacapo_c1_pruned silently yielded no translator for those
    five, which made the dump skip them, which made the simulator substitute oracle
    times for them -- reading as a near-perfect model result rather than an error.
    """
    if bench_name.startswith('dacapo_'):
        return 'dacapo_c1'
    m = _SPEC_NUM_RE.match(bench_name)
    if m:
        return 'spec_2026' if int(m.group(1)) >= 700 else 'spec_2017'
    return None


def _bench_suite_dir(bench_name):
    """Suite subdirectory in the cross-processor model tree."""
    if bench_name.startswith('dacapo_'):
        return 'dacapo_c1'
    m = _SPEC_NUM_RE.match(bench_name)
    if m:
        return 'spec_2026' if int(m.group(1)) >= 700 else 'spec_2017'
    return None


# Feature set / model variant used by the forecaster's translation step. Default 'top4'
# reproduces the per-workload LOOCV models. Set CBM_FEATURE_SET=general_temporal or
# general_insample to translate through the single no-holdout general model instead, so the
# forecast can be evaluated at each cross-processor / cross-frequency model quality.
CBM_FEATURE_SET = os.environ.get('CBM_FEATURE_SET', 'top4')


def _cbm_model_file(bench_name, feature_set=None):
    """Model filename for a translation pair. Per-workload LOOCV uses model_{bench}.cbm;
    a general/no-holdout variant is a single model_general_{feature_set}.cbm."""
    fs = feature_set if feature_set is not None else CBM_FEATURE_SET
    if fs.startswith('general'):
        return f'model_general_{fs}.cbm'
    return f'model_{bench_name}.cbm'


def _load_cbm(cbm_model_dir, src_freq, tgt_freq, bench_name):
    """Load a CatBoost cross-frequency translation model, or None if not found.
    cbm_model_dir is the per-cpu root containing
    {suite}/{feature_set}/{src}GHz_to_{tgt}GHz/model_*.cbm."""
    suite = _bench_suite_dir_cross_freq(bench_name)
    if suite is None:
        return None
    path = os.path.join(
        cbm_model_dir, suite, CBM_FEATURE_SET,
        f'{src_freq}GHz_to_{tgt_freq}GHz', _cbm_model_file(bench_name)
    )
    if not os.path.exists(path):
        return None
    m = _CatBoost()
    m.load_model(path)
    return m


def _load_cbm_cross_proc(cbm_cross_proc_dir, src_cpu, src_freq, tgt_cpu, tgt_freq, bench_name):
    """Load a CatBoost cross-processor translation model, or None if not found."""
    suite = _bench_suite_dir(bench_name)
    if suite is None:
        return None
    path = os.path.join(
        cbm_cross_proc_dir,
        f'cpu{src_cpu}_to_cpu{tgt_cpu}', suite, CBM_FEATURE_SET,
        f'cpu{src_cpu}_{src_freq}GHz_to_cpu{tgt_cpu}_{tgt_freq}GHz',
        _cbm_model_file(bench_name)
    )
    if not os.path.exists(path):
        return None
    m = _CatBoost()
    m.load_model(path)
    return m


def _translate_ref_cycles(cbm, full_df, features=None):
    """
    Apply a CatBoost model to translate the ref_cycles column in full_df.
    features: explicit column names, or None to auto-discover from cbm.feature_names_.
    Returns int64 array or None if any required feature is missing from full_df.
    """
    if features is None:
        features = list(cbm.feature_names_)
    if not all(f in full_df.columns for f in features):
        return None
    X = full_df[features]
    src_ref = full_df['ref_cycles'].values
    mask = src_ref > 0
    pred_log_ratio = cbm.predict(X)
    translated = np.where(mask, np.exp(pred_log_ratio) * src_ref, src_ref)
    return translated.round().astype(np.int64)


def _bench_suite_dir_ct(bench_name):
    """Suite label used in the counter_translation tree (spec2017/spec2026/dacapo)."""
    if bench_name.startswith('dacapo'):
        return 'dacapo'
    m = _SPEC_NUM_RE.match(bench_name)
    if m:
        return 'spec2026' if int(m.group(1)) >= 700 else 'spec2017'
    return None


def _load_cbm_counter(counter_dir, counter, src_cpu, src_freq, tgt_cpu, tgt_freq, bench_name):
    """Load a per-counter (e.g. cpu_cycles) cross-proc translator from the
    counter_translation tree, or None if not found."""
    suite = _bench_suite_dir_ct(bench_name)
    if suite is None:
        return None
    path = os.path.join(
        counter_dir, f'cpu{src_cpu}_to_cpu{tgt_cpu}', suite, counter, 'top4',
        f'cpu{src_cpu}_{src_freq}GHz_to_cpu{tgt_cpu}_{tgt_freq}GHz',
        f'model_{bench_name}.cbm'
    )
    if not os.path.exists(path):
        return None
    m = _CatBoost()
    m.load_model(path)
    return m


def _translate_counter(cbm, full_df, counter):
    """translated = exp(predicted log-ratio) * source[counter]; int64 array or None."""
    features = list(cbm.feature_names_)
    if not all(f in full_df.columns for f in features):
        return None
    src = full_df[counter].values
    mask = src > 0
    pred_log_ratio = cbm.predict(full_df[features])
    return np.where(mask, np.exp(pred_log_ratio) * src, src).round().astype(np.int64)


def apply_heterogeneous_history(args, train_data):
    """
    With probability args.heterogeneous_prob, replace each row of the
    training-set with the same-position row from a randomly selected donor
    trace (a different frequency and/or core, per args.heterogeneous_mode).
    Selection and donor assignment are seeded by args.heterogeneous_seed for
    repeatability. The test set is never touched (caller only passes train_data).

    When args.cbm_model_dir (cross_freq) or args.cbm_cross_proc_dir (cross_proc /
    cross_proc_freq) is set, ref_cycles is translated to the target operating point
    via a pre-trained CatBoost model before injection; the other counters are copied
    directly. Falls back to naive swap when no model exists for a given workload.

    If args.add_heterogeneity_features is set, also appends het_flag and
    het_source_freq columns marking which samples were replaced and which
    frequency their donor came from.
    """
    add_features = getattr(args, 'add_heterogeneity_features', False)
    if add_features:
        train_data = add_heterogeneity_columns(args, train_data)

    if getattr(args, 'heterogeneous_prob', 0.0) <= 0:
        return train_data

    counter_cols = [c for c in train_data.columns if c not in ('het_flag', 'het_source_freq')]
    donors = _find_donor_files(args, counter_cols)
    if not donors:
        return train_data

    cbm_model_dir      = getattr(args, 'cbm_model_dir', None)
    cbm_cross_proc_dir = getattr(args, 'cbm_cross_proc_dir', None)
    cbm_counter_dir    = getattr(args, 'cbm_cross_proc_counter_dir', None)
    mode = args.heterogeneous_mode

    use_translation = (
        cbm_model_dir is not None
        and mode == 'cross_freq'
        and _CATBOOST_AVAILABLE
    )
    use_cross_proc_translation = (
        cbm_cross_proc_dir is not None
        and mode in ('cross_proc', 'cross_proc_freq')
        and _CATBOOST_AVAILABLE
    )

    if use_translation or use_cross_proc_translation:
        # Only keep donors from allowed suites (dacapo_c1, spec_2017, spec_2026); exclude dacapo_c2
        donors = [
            d for d in donors
            if any(f'/{s}/' in d for s in _ALLOWED_DONOR_SUITES)
        ]
        if not donors:
            return train_data

    tgt_freq = _own_freq(args)
    tgt_cpu  = _own_cpu(args)

    rng = np.random.default_rng(args.heterogeneous_seed)
    n = train_data.shape[0]
    swap_mask = rng.random(n) < args.heterogeneous_prob
    swap_indices = np.where(swap_mask)[0]
    if len(swap_indices) == 0:
        return train_data

    train_data = train_data.copy()

    # donor_cache: path -> (full_df, src_freq, src_cpu, translated_ref_or_None)
    donor_cache = {}
    cbm_cache = {}

    for idx in swap_indices:
        donor_path = donors[rng.integers(len(donors))]
        if donor_path not in donor_cache:
            full_df = _crop(repair_ref_cycles_df(pd.read_csv(donor_path)), args).reset_index(drop=True)
            dm = BENCHMARK_NAME_RE.match(os.path.basename(donor_path).replace('.csv', ''))
            src_freq   = float(dm.group('freq'))
            src_cpu    = dm.group('cpu')
            bench_name = dm.group('rest')

            translated_ref = None
            translated_cpu = None

            if use_translation:
                cbm_key = (src_freq, tgt_freq, bench_name)
                if cbm_key not in cbm_cache:
                    cbm_cache[cbm_key] = _load_cbm(cbm_model_dir, src_freq, tgt_freq, bench_name)
                cbm = cbm_cache[cbm_key]
                if cbm is not None:
                    translated_ref = _translate_ref_cycles(cbm, full_df)
            elif use_cross_proc_translation:
                cbm_key = (src_cpu, src_freq, tgt_cpu, tgt_freq, bench_name)
                if cbm_key not in cbm_cache:
                    cbm_cache[cbm_key] = _load_cbm_cross_proc(
                        cbm_cross_proc_dir, src_cpu, src_freq, tgt_cpu, tgt_freq, bench_name
                    )
                cbm = cbm_cache[cbm_key]
                if cbm is not None:
                    translated_ref = _translate_ref_cycles(cbm, full_df)
                # optionally also translate cpu_cycles across cores (freq-invariant,
                # so only meaningful when the core changes)
                if cbm_counter_dir is not None and src_cpu != tgt_cpu:
                    ck = ('cpu_cycles', src_cpu, src_freq, tgt_cpu, tgt_freq, bench_name)
                    if ck not in cbm_cache:
                        cbm_cache[ck] = _load_cbm_counter(
                            cbm_counter_dir, 'cpu_cycles', src_cpu, src_freq, tgt_cpu, tgt_freq, bench_name
                        )
                    ccbm = cbm_cache[ck]
                    if ccbm is not None:
                        translated_cpu = _translate_counter(ccbm, full_df, 'cpu_cycles')

            donor_cache[donor_path] = (full_df, src_freq, src_cpu, translated_ref, translated_cpu)

        full_df, donor_freq, src_cpu, translated_ref, translated_cpu = donor_cache[donor_path]
        row_label = train_data.index[idx]
        row_i     = idx % len(full_df)
        donor_row = full_df.iloc[row_i]

        for col in counter_cols:
            if col in full_df.columns:
                train_data.loc[row_label, col] = donor_row[col]

        if translated_ref is not None and 'ref_cycles' in counter_cols:
            train_data.loc[row_label, 'ref_cycles'] = translated_ref[row_i]
        if translated_cpu is not None and 'cpu_cycles' in counter_cols:
            train_data.loc[row_label, 'cpu_cycles'] = translated_cpu[row_i]

        if add_features:
            train_data.loc[row_label, 'het_flag'] = 1
            train_data.loc[row_label, 'het_source_freq'] = donor_freq

    return train_data


def reshape_with_batch_size(X_train, y_train, X_test, y_test, batch_size):
    """
    Truncates the ends of the datasets to ensure they are perfectly divisible by the batch size.
    """
    train_drop = get_drop_samples(X_train.shape[0], batch_size)
    if train_drop != 0:
        X_train = X_train.iloc[:-train_drop, :]
        y_train = y_train.iloc[:-train_drop]

    test_drop = get_drop_samples(X_test.shape[0], batch_size)
    if test_drop != 0:
        X_test = X_test.iloc[:-test_drop, :]
        y_test = y_test.iloc[:-test_drop]

    return X_train, y_train, X_test, y_test

def get_separate_time_series_splits(args, data):
    """
    Splits the data into train/test FIRST, then applies time series transforms independently
    to prevent data leakage across the train/test boundary.
    """
    split_idx = int(data.shape[0] * args.train_size / 100)
    train_data = data.iloc[:split_idx, :]
    test_data = data.iloc[split_idx:, :]

    train_data = apply_heterogeneous_history(args, train_data)

    if getattr(args, 'add_heterogeneity_features', False):
        test_data = add_heterogeneity_columns(args, test_data)

    train_timeseries = get_transformed_time_series(args, train_data)
    X_train, y_train, _, _ = get_split_data_set(args, train_timeseries)

    test_timeseries = get_transformed_time_series(args, test_data)
    _, _, X_test, y_test = get_split_data_set(args, test_timeseries)

    # Populate attributes for downstream compatibility (evaluate.py expects them)
    train_timeseries.X_train = X_train
    train_timeseries.y_train = y_train
    train_timeseries.X_test = X_test
    train_timeseries.y_test = y_test

    test_timeseries.X_train = X_train
    test_timeseries.y_train = y_train
    test_timeseries.X_test = X_test
    test_timeseries.y_test = y_test

    return X_train, y_train, X_test, y_test, train_timeseries, test_timeseries

def get_transformed_time_series(args, data):
    timeseries = ts.TimeSeriesTransforms(data)
    timeseries.set_transforms(args)
    return timeseries

def get_split_data_set(args, timeseries):
    timeseries.set_train_test_split(args)
    return timeseries.X_train, timeseries.y_train, timeseries.X_test, timeseries.y_test

def get_drop_samples(data_shape, batch_size):
    assert data_shape >= batch_size, "Batch size is too large for the provided data"
    batch_count = int(data_shape / batch_size)
    drop_samples = 0
    if batch_size > 1:
        drop_samples = data_shape - (batch_count * batch_size)
    return drop_samples

import os
import re
import glob
import numpy as np
import pandas as pd
import src.TimeSeries as ts

BENCHMARK_NAME_RE = re.compile(
    r"^aligned_(?P<rest>.+)_(?P<freq>[\d.]+)GHz_cpu(?P<cpu>\d+)_phase(?P<phase>\d+)$"
)

def get_raw_data(args):
    """
    Loads the pre-aligned and pre-processed CSV directly from the processed_data folder.
    Bypasses legacy traces.py and hardcoded frequency hacks.
    """
    # 1. Dynamically find the PerfCount root directory (4 folders up from src/create_dataset.py)
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
    
    # 2. Build the exact path to our new processed data
    file_path = os.path.join(root_dir, 'processed_data_10M', args.dataset, f"{args.benchmark}.csv")
    
    if not os.path.exists(file_path):
        print(f"[ERROR] Could not find dataset at: {file_path}")
        return None
        
    # 3. Load the data
    df = pd.read_csv(file_path)
    
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

    Only donors whose header contains every column in `columns` are kept.
    """
    m = BENCHMARK_NAME_RE.match(args.benchmark)
    if not m:
        return []

    rest, cur_freq, cur_cpu, phase = m.group('rest'), m.group('freq'), m.group('cpu'), m.group('phase')
    data_dir = _get_data_dir(args)

    pattern = os.path.join(data_dir, f"aligned_{rest}_*GHz_cpu*_phase{phase}.csv")
    donors = []
    for fpath in glob.glob(pattern):
        name = os.path.basename(fpath).replace('.csv', '')
        dm = BENCHMARK_NAME_RE.match(name)
        if not dm:
            continue
        d_freq, d_cpu = dm.group('freq'), dm.group('cpu')

        if args.heterogeneous_mode == 'cross_freq':
            if d_cpu != cur_cpu or d_freq == cur_freq:
                continue
        else:  # cross_proc
            if d_cpu == cur_cpu:
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
    m = BENCHMARK_NAME_RE.match(args.benchmark)
    return float(m.group('freq')) if m else 0.0


def add_heterogeneity_columns(args, data):
    """
    Append het_flag (0/1) and het_source_freq (GHz) columns, defaulting to
    "not heterogeneous, sourced from this trace's own frequency".
    """
    data = data.copy()
    data['het_flag'] = 0
    data['het_source_freq'] = _own_freq(args)
    return data


def apply_heterogeneous_history(args, train_data):
    """
    With probability args.heterogeneous_prob, replace each row of the
    training-set with the same-position row from a randomly selected donor
    trace (a different frequency and/or core, per args.heterogeneous_mode).
    Selection and donor assignment are seeded by args.heterogeneous_seed for
    repeatability. The test set is never touched (caller only passes train_data).

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

    rng = np.random.default_rng(args.heterogeneous_seed)
    n = train_data.shape[0]
    swap_mask = rng.random(n) < args.heterogeneous_prob
    swap_indices = np.where(swap_mask)[0]
    if len(swap_indices) == 0:
        return train_data

    train_data = train_data.copy()

    donor_cache = {}
    for idx in swap_indices:
        donor_path = donors[rng.integers(len(donors))]
        if donor_path not in donor_cache:
            donor_df = _crop(pd.read_csv(donor_path)[counter_cols], args).reset_index(drop=True)
            dm = BENCHMARK_NAME_RE.match(os.path.basename(donor_path).replace('.csv', ''))
            donor_cache[donor_path] = (donor_df, float(dm.group('freq')))
        donor_df, donor_freq = donor_cache[donor_path]
        if idx >= len(donor_df):
            continue
        row_label = train_data.index[idx]
        train_data.loc[row_label, counter_cols] = donor_df.iloc[idx].values
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
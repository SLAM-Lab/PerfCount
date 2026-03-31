import os
import pandas as pd
import src.TimeSeries as ts

def get_raw_data(args):
    """
    Loads the pre-aligned and pre-processed CSV directly from the processed_data folder.
    Bypasses legacy traces.py and hardcoded frequency hacks.
    """
    # 1. Dynamically find the PerfCount root directory (4 folders up from src/create_dataset.py)
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
    
    # 2. Build the exact path to our new processed data
    file_path = os.path.join(root_dir, 'processed_data', args.dataset, f"{args.benchmark}.csv")
    
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
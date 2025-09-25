"""
Data processing module for cross-platform prediction evaluation.

This module handles data loading, alignment, preprocessing, and feature-target
pair creation.
"""

import pandas as pd
import numpy as np
import os
import glob
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import IsolationForest


def read_csv_to_dataframe(csv_path):
    """
    Read a CSV file and convert it to a pandas DataFrame
    
    Args:
        csv_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: DataFrame with counter names as columns and samples as
            rows
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"Loaded CSV from {csv_path}: {df.shape[0]} rows, "
          f"{df.shape[1]} columns")
    return df


def align_dataframes(source_df, target_df):
    """
    Align two DataFrames to the same length by truncating the longer one
    
    Args:
        source_df (pd.DataFrame): Source DataFrame
        target_df (pd.DataFrame): Target DataFrame
        
    Returns:
        tuple: (aligned_source_df, aligned_target_df)
    """
    source_len = len(source_df)
    target_len = len(target_df)
    
    print(f"Source DataFrame length: {source_len}")
    print(f"Target DataFrame length: {target_len}")
    
    if source_len > target_len:
        print(f"Truncating source DataFrame from {source_len} to "
              f"{target_len} rows")
        source_df = source_df.iloc[:target_len]
    elif target_len > source_len:
        print(f"Truncating target DataFrame from {target_len} to "
              f"{source_len} rows")
        target_df = target_df.iloc[:source_len]
    else:
        print("DataFrames already have the same length")
    
    print(f"Aligned DataFrames: {len(source_df)} rows each")
    return source_df, target_df


def preprocess_data(X, y, remove_outliers=True, feature_selection=None,
                    scaler_type='standard'):
    """
    Preprocess data with outlier removal, feature selection, and scaling
    
    Args:
        X (np.ndarray): Feature matrix
        y (np.ndarray): Target values
        remove_outliers (bool): Whether to remove outliers
        feature_selection (int or None): Number of features to select
            (None for all)
        scaler_type (str): Type of scaler ('standard', 'robust')
        
    Returns:
        tuple: (X_processed, y_processed, feature_names, scaler,
            feature_selector)
    """
    X_processed = X.copy()
    y_processed = y.copy()
    
    # Remove outliers using Isolation Forest
    if remove_outliers:
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        outlier_mask = iso_forest.fit_predict(X_processed) == 1
        X_processed = X_processed[outlier_mask]
        y_processed = y_processed[outlier_mask]
        print(f"  Removed {len(X) - len(X_processed)} outliers "
              f"({100 * (len(X) - len(X_processed)) / len(X):.1f}%)")
    
    # Feature selection
    feature_selector = None
    if (feature_selection is not None and
            feature_selection < X_processed.shape[1]):
        feature_selector = SelectKBest(
            score_func=f_regression, k=feature_selection
        )
        X_processed = feature_selector.fit_transform(X_processed, y_processed)
        print(f"  Selected {feature_selection} best features from {X.shape[1]}")
    
    # Scaling
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown scaler type: {scaler_type}")
    
    X_processed = scaler.fit_transform(X_processed)
    
    return X_processed, y_processed, None, scaler, feature_selector


def create_feature_target_pairs(source_df, target_df, target_column,
                                preprocess=True, **preprocess_kwargs):
    """
    Create source-target pairs for ML training with optional preprocessing
    
    Args:
        source_df (pd.DataFrame): Source features DataFrame
        target_df (pd.DataFrame): Target DataFrame
        target_column (str): Name of the target column to predict
        preprocess (bool): Whether to apply preprocessing
        **preprocess_kwargs: Additional arguments for preprocessing
        
    Returns:
        tuple: (X_features, y_targets, feature_names, scaler, feature_selector)
    """
    if target_column not in target_df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in target DataFrame. "
            f"Available columns: {list(target_df.columns)}"
        )
    
    # Use all columns from source as features
    X = source_df.values
    y = target_df[target_column].values
    feature_names = list(source_df.columns)
    
    print(f"Created feature-target pairs:")
    print(f"  Features shape: {X.shape}")
    print(f"  Target shape: {y.shape}")
    print(f"  Target column: {target_column}")
    
    if preprocess:
        X, y, _, scaler, feature_selector = preprocess_data(
            X, y, **preprocess_kwargs
        )
        return X, y, feature_names, scaler, feature_selector
    else:
        return X, y, feature_names, None, None


def load_workload_data(source_dir, target_dir, target_column):
    """
    Load and align all workload data from source and target directories
    
    Args:
        source_dir (str): Directory containing source CSV files
        target_dir (str): Directory containing target CSV files
        target_column (str): Name of the target column to predict
        
    Returns:
        dict: Dictionary mapping workload names to (source_df, target_df) tuples
    """
    workload_data = {}
    
    # Get all CSV files from both directories
    source_files = glob.glob(os.path.join(source_dir, "*.csv"))
    target_files = glob.glob(os.path.join(target_dir, "*.csv"))
    
    print(f"Found {len(source_files)} source files and "
          f"{len(target_files)} target files")
    
    # Extract workload names (assuming filenames match between source and target)
    source_workloads = {
        os.path.basename(f).replace('.csv', ''): f for f in source_files
    }
    target_workloads = {
        os.path.basename(f).replace('.csv', ''): f for f in target_files
    }
    
    # Find common workloads
    common_workloads = (set(source_workloads.keys()) &
                        set(target_workloads.keys()))
    print(f"Found {len(common_workloads)} common workloads: "
          f"{sorted(common_workloads)}")
    
    for workload in common_workloads:
        print(f"\nProcessing workload: {workload}")
        
        # Read and align data for this workload
        source_df = read_csv_to_dataframe(source_workloads[workload])
        target_df = read_csv_to_dataframe(target_workloads[workload])
        
        # Align DataFrames to same length
        source_df, target_df = align_dataframes(source_df, target_df)
        
        workload_data[workload] = (source_df, target_df)
    
    return workload_data


def combine_workload_data(workload_data, target_column):
    """
    Combine multiple workloads into single training dataset
    
    Args:
        workload_data (dict): Dictionary mapping workload names to
            (source_df, target_df) tuples
        target_column (str): Name of the target column to predict
        
    Returns:
        tuple: (X_combined, y_combined) combined features and targets
    """
    all_X = []
    all_y = []
    
    for workload, (source_df, target_df) in workload_data.items():
        X, y, _, _, _ = create_feature_target_pairs(
            source_df, target_df, target_column, preprocess=False
        )
        all_X.append(X)
        all_y.append(y)
    
    # Combine all data
    X_combined = np.vstack(all_X)
    y_combined = np.concatenate(all_y)
    
    print(f"Combined dataset shape: {X_combined.shape}, "
          f"targets: {y_combined.shape}")
    return X_combined, y_combined

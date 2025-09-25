# Cross-platform prediction evaluation script
# This file is for evaluating performance predictions across different platforms

import pandas as pd
import numpy as np
import argparse
import os
import glob
import json
import pickle
import time
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor, BaggingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge, HuberRegressor, SGDRegressor, PassiveAggressiveRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV, HalvingRandomSearchCV
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.ensemble import IsolationForest
from scipy import stats
from scipy.stats import wilcoxon, ttest_rel
import warnings
warnings.filterwarnings('ignore')

# Try to import additional boosting libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("Warning: CatBoost not available. Install with: pip install catboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM not available. Install with: pip install lightgbm")

# Configuration
DEFAULT_OUTPUT_DIR = "results"
DEFAULT_MODELS_DIR = "models"
DEFAULT_PLOTS_DIR = "plots"

def create_output_directories(output_dir=DEFAULT_OUTPUT_DIR):
    """Create output directories for results, models, and plots"""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, DEFAULT_MODELS_DIR), exist_ok=True)
    os.makedirs(os.path.join(output_dir, DEFAULT_PLOTS_DIR), exist_ok=True)
    return output_dir

def read_csv_to_dataframe(csv_path):
    """
    Read a CSV file and convert it to a pandas DataFrame
    
    Args:
        csv_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: DataFrame with counter names as columns and samples as rows
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"Loaded CSV from {csv_path}: {df.shape[0]} rows, {df.shape[1]} columns")
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
        print(f"Truncating source DataFrame from {source_len} to {target_len} rows")
        source_df = source_df.iloc[:target_len]
    elif target_len > source_len:
        print(f"Truncating target DataFrame from {target_len} to {source_len} rows")
        target_df = target_df.iloc[:source_len]
    else:
        print("DataFrames already have the same length")
    
    print(f"Aligned DataFrames: {len(source_df)} rows each")
    return source_df, target_df

def preprocess_data(X, y, remove_outliers=True, feature_selection=None, scaler_type='standard'):
    """
    Preprocess data with outlier removal, feature selection, and scaling
    
    Args:
        X (np.ndarray): Feature matrix
        y (np.ndarray): Target values
        remove_outliers (bool): Whether to remove outliers
        feature_selection (int or None): Number of features to select (None for all)
        scaler_type (str): Type of scaler ('standard', 'robust')
        
    Returns:
        tuple: (X_processed, y_processed, feature_names, scaler, feature_selector)
    """
    X_processed = X.copy()
    y_processed = y.copy()
    
    # Remove outliers using Isolation Forest
    if remove_outliers:
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        outlier_mask = iso_forest.fit_predict(X_processed) == 1
        X_processed = X_processed[outlier_mask]
        y_processed = y_processed[outlier_mask]
        print(f"  Removed {len(X) - len(X_processed)} outliers ({100 * (len(X) - len(X_processed)) / len(X):.1f}%)")
    
    # Feature selection
    feature_selector = None
    if feature_selection is not None and feature_selection < X_processed.shape[1]:
        feature_selector = SelectKBest(score_func=f_regression, k=feature_selection)
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

def create_feature_target_pairs(source_df, target_df, target_column, preprocess=True, **preprocess_kwargs):
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
        raise ValueError(f"Target column '{target_column}' not found in target DataFrame. Available columns: {list(target_df.columns)}")
    
    # Use all columns from source as features
    X = source_df.values
    y = target_df[target_column].values
    feature_names = list(source_df.columns)
    
    print(f"Created feature-target pairs:")
    print(f"  Features shape: {X.shape}")
    print(f"  Target shape: {y.shape}")
    print(f"  Target column: {target_column}")
    
    if preprocess:
        X, y, _, scaler, feature_selector = preprocess_data(X, y, **preprocess_kwargs)
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
    
    print(f"Found {len(source_files)} source files and {len(target_files)} target files")
    
    # Extract workload names (assuming filenames match between source and target)
    source_workloads = {os.path.basename(f).replace('.csv', ''): f for f in source_files}
    target_workloads = {os.path.basename(f).replace('.csv', ''): f for f in target_files}
    
    # Find common workloads
    common_workloads = set(source_workloads.keys()) & set(target_workloads.keys())
    print(f"Found {len(common_workloads)} common workloads: {sorted(common_workloads)}")
    
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
        workload_data (dict): Dictionary mapping workload names to (source_df, target_df) tuples
        target_column (str): Name of the target column to predict
        
    Returns:
        tuple: (X_combined, y_combined) combined features and targets
    """
    all_X = []
    all_y = []
    
    for workload, (source_df, target_df) in workload_data.items():
        X, y = create_feature_target_pairs(source_df, target_df, target_column)
        all_X.append(X)
        all_y.append(y)
    
    # Combine all data
    X_combined = np.vstack(all_X)
    y_combined = np.concatenate(all_y)
    
    print(f"Combined dataset shape: {X_combined.shape}, targets: {y_combined.shape}")
    return X_combined, y_combined

def get_model_configurations():
    """
    Get all regression models with their hyperparameter grids
    
    Returns:
        dict: Dictionary mapping model names to (model_class, param_grid) tuples
    """
    models = {}
    
    # Linear Models
    models['LinearRegression'] = (LinearRegression(), {})
    
    models['Ridge'] = (Ridge(random_state=42), {
        'alpha': [0.1, 1.0, 10.0, 100.0, 1000.0]
    })
    
    models['Lasso'] = (Lasso(random_state=42), {
        'alpha': [0.01, 0.1, 1.0, 10.0]
    })
    
    models['ElasticNet'] = (ElasticNet(random_state=42), {
        'alpha': [0.01, 0.1, 1.0, 10.0],
        'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
    })
    
    models['BayesianRidge'] = (BayesianRidge(), {
        'alpha_1': [1e-6, 1e-5, 1e-4],
        'alpha_2': [1e-6, 1e-5, 1e-4],
        'lambda_1': [1e-6, 1e-5, 1e-4],
        'lambda_2': [1e-6, 1e-5, 1e-4]
    })
    
    models['HuberRegressor'] = (HuberRegressor(), {
        'epsilon': [1.1, 1.35, 1.5, 2.0],
        'alpha': [0.0001, 0.001, 0.01, 0.1]
    })
    
    models['SGDRegressor'] = (SGDRegressor(random_state=42), {
        'loss': ['squared_error', 'huber', 'epsilon_insensitive'],
        'penalty': ['l2', 'l1', 'elasticnet'],
        'alpha': [0.0001, 0.001, 0.01, 0.1],
        'learning_rate': ['constant', 'optimal', 'invscaling']
    })
    
    models['PassiveAggressiveRegressor'] = (PassiveAggressiveRegressor(random_state=42), {
        'C': [0.01, 0.1, 1.0, 10.0],
        'epsilon': [0.01, 0.1, 0.5, 1.0]
    })
    
    # Support Vector Machines
    models['SVR'] = (SVR(), {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
        'kernel': ['rbf', 'linear', 'poly']
    })
    
    models['LinearSVR'] = (LinearSVR(random_state=42), {
        'C': [0.1, 1, 10, 100],
        'epsilon': [0.01, 0.1, 0.5, 1.0],
        'loss': ['epsilon_insensitive', 'squared_epsilon_insensitive']
    })
    
    # Neighbors
    models['KNeighborsRegressor'] = (KNeighborsRegressor(), {
        'n_neighbors': [3, 5, 7, 9, 11, 15],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree']
    })
    
    models['RadiusNeighborsRegressor'] = (RadiusNeighborsRegressor(), {
        'radius': [1.0, 2.0, 5.0, 10.0],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree']
    })
    
    # Trees
    models['DecisionTreeRegressor'] = (DecisionTreeRegressor(random_state=42), {
        'max_depth': [None, 5, 10, 15, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['squared_error', 'friedman_mse', 'absolute_error']
    })
    
    models['ExtraTreeRegressor'] = (ExtraTreeRegressor(random_state=42), {
        'max_depth': [None, 5, 10, 15, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    })
    
    # Ensemble Methods
    models['RandomForestRegressor'] = (RandomForestRegressor(random_state=42), {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['squared_error', 'absolute_error']
    })
    
    models['ExtraTreesRegressor'] = (ExtraTreesRegressor(random_state=42), {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    })
    
    models['GradientBoostingRegressor'] = (GradientBoostingRegressor(random_state=42), {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7, 10],
        'subsample': [0.8, 0.9, 1.0],
        'criterion': ['squared_error', 'friedman_mse']
    })
    
    models['AdaBoostRegressor'] = (AdaBoostRegressor(random_state=42), {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.5, 1.0],
        'loss': ['linear', 'square', 'exponential']
    })
    
    models['BaggingRegressor'] = (BaggingRegressor(random_state=42), {
        'n_estimators': [10, 50, 100],
        'max_samples': [0.5, 0.7, 0.9, 1.0],
        'max_features': [0.5, 0.7, 0.9, 1.0]
    })
    
    # Neural Networks
    models['MLPRegressor'] = (MLPRegressor(random_state=42, max_iter=1000), {
        'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (100, 100)],
        'activation': ['relu', 'tanh', 'logistic'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'adaptive'],
        'solver': ['adam', 'lbfgs']
    })
    
    # Gaussian Process
    models['GaussianProcessRegressor'] = (GaussianProcessRegressor(random_state=42), {
        'kernel': [RBF(), C(1.0) * RBF(), C(1.0) * RBF() + C(1.0)],
        'alpha': [1e-10, 1e-8, 1e-6, 1e-4]
    })
    
    # Add XGBoost if available
    if XGBOOST_AVAILABLE:
        models['XGBRegressor'] = (xgb.XGBRegressor(random_state=42, verbosity=0), {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7, 10],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 1],
            'reg_lambda': [1, 1.5, 2]
        })
    
    # Add CatBoost if available
    if CATBOOST_AVAILABLE:
        models['CatBoostRegressor'] = (cb.CatBoostRegressor(random_seed=42, verbose=False), {
            'iterations': [50, 100, 200],
            'depth': [3, 5, 7, 10],
            'learning_rate': [0.01, 0.1, 0.2],
            'l2_leaf_reg': [1, 3, 5, 7, 9]
        })
    
    # Add LightGBM if available
    if LIGHTGBM_AVAILABLE:
        models['LGBMRegressor'] = (lgb.LGBMRegressor(random_state=42, verbosity=-1), {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7, 10],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 1],
            'reg_lambda': [0, 0.1, 1]
        })
    
    return models

def train_and_tune_model(model_name, model, param_grid, X_train, y_train, cv_folds=3, search_strategy='auto', max_iter=50):
    """
    Train and hyperparameter tune a regression model with advanced search strategies
    
    Args:
        model_name (str): Name of the model
        model: Model instance
        param_grid (dict): Hyperparameter grid
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training targets
        cv_folds (int): Number of CV folds for hyperparameter tuning
        search_strategy (str): Search strategy ('auto', 'grid', 'random', 'halving_grid', 'halving_random')
        max_iter (int): Maximum iterations for random search
        
    Returns:
        tuple: (best_model, best_params, best_score)
    """
    print(f"  Training and tuning {model_name}...")
    
    # Scale features for models that benefit from it
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Calculate total parameter combinations
    total_combinations = 1
    for param_values in param_grid.values():
        total_combinations *= len(param_values)
    
    print(f"    Parameter space: {total_combinations} total combinations")
    
    # Choose search strategy
    if len(param_grid) == 0:
        # No hyperparameters to tune
        model.fit(X_train_scaled, y_train)
        return model, {}, 0.0
    
    elif search_strategy == 'auto':
        # Automatic strategy selection based on parameter space size
        if total_combinations <= 20:
            search_strategy = 'grid'
        elif total_combinations <= 100:
            search_strategy = 'halving_grid'
        else:
            search_strategy = 'halving_random'
    
    # Create search object based on strategy
    if search_strategy == 'grid':
        print(f"    Using GridSearchCV (exhaustive search)")
        search = GridSearchCV(
            model, param_grid, cv=cv_folds,
            scoring='neg_mean_absolute_percentage_error',
            n_jobs=-1, verbose=0
        )
        
    elif search_strategy == 'random':
        print(f"    Using RandomizedSearchCV (random sampling)")
        search = RandomizedSearchCV(
            model, param_grid, n_iter=min(max_iter, total_combinations), 
            cv=cv_folds, scoring='neg_mean_absolute_percentage_error', 
            random_state=42, n_jobs=-1, verbose=0
        )
        
    elif search_strategy == 'halving_grid':
        print(f"    Using HalvingGridSearchCV (successive halving)")
        search = HalvingGridSearchCV(
            model, param_grid, cv=cv_folds,
            scoring='neg_mean_absolute_percentage_error',
            n_jobs=-1, verbose=0, random_state=42
        )
        
    elif search_strategy == 'halving_random':
        print(f"    Using HalvingRandomSearchCV (successive halving + random)")
        search = HalvingRandomSearchCV(
            model, param_grid, cv=cv_folds,
            scoring='neg_mean_absolute_percentage_error',
            n_jobs=-1, verbose=0, random_state=42,
            n_candidates=min(max_iter, total_combinations)
        )
    
    else:
        raise ValueError(f"Unknown search strategy: {search_strategy}")
    
    # Perform the search
    search.fit(X_train_scaled, y_train)
    
    print(f"    Best CV score: {-search.best_score_:.4f}")
    print(f"    Best params: {search.best_params_}")
    
    return search.best_estimator_, search.best_params_, -search.best_score_

def train_single_model(args):
    """
    Train a single model (for parallel processing)
    
    Args:
        args: Tuple of (model_name, model, param_grid, X_train, y_train, cv_folds, search_strategy, max_iter)
        
    Returns:
        tuple: (model_name, best_model, best_params, best_score) or (model_name, None, None, None) if failed
    """
    model_name, model, param_grid, X_train, y_train, cv_folds, search_strategy, max_iter = args
    
    try:
        best_model, best_params, best_score = train_and_tune_model(
            model_name, model, param_grid, X_train, y_train, cv_folds, search_strategy, max_iter
        )
        return model_name, best_model, best_params, best_score
    except Exception as e:
        print(f"    Error training {model_name}: {str(e)}")
        return model_name, None, None, None

def train_all_models(X_train, y_train, cv_folds=3, search_strategy='auto', max_iter=50, n_jobs=None):
    """
    Train and tune all regression models with advanced hyperparameter search and parallel processing
    
    Args:
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training targets
        cv_folds (int): Number of CV folds for hyperparameter tuning
        search_strategy (str): Search strategy ('auto', 'grid', 'random', 'halving_grid', 'halving_random')
        max_iter (int): Maximum iterations for random search
        n_jobs (int): Number of parallel jobs (None for auto)
        
    Returns:
        dict: Dictionary mapping model names to (model, params, cv_score) tuples
    """
    print(f"Training and tuning all models on {X_train.shape[0]} samples with {X_train.shape[1]} features")
    print(f"Search strategy: {search_strategy}, Max iterations: {max_iter}")
    
    models = get_model_configurations()
    
    if n_jobs is None:
        n_jobs = min(cpu_count(), len(models))
    
    print(f"Using {n_jobs} parallel jobs")
    
    # Prepare arguments for parallel processing
    model_args = [
        (model_name, model, param_grid, X_train, y_train, cv_folds, search_strategy, max_iter)
        for model_name, (model, param_grid) in models.items()
    ]
    
    trained_models = {}
    
    if n_jobs == 1:
        # Sequential processing
        for args in model_args:
            model_name, best_model, best_params, best_score = train_single_model(args)
            if best_model is not None:
                trained_models[model_name] = (best_model, best_params, best_score)
    else:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            future_to_model = {executor.submit(train_single_model, args): args[0] for args in model_args}
            
            for future in as_completed(future_to_model):
                model_name, best_model, best_params, best_score = future.result()
                if best_model is not None:
                    trained_models[model_name] = (best_model, best_params, best_score)
    
    print(f"Successfully trained {len(trained_models)} models")
    return trained_models

def evaluate_model_comprehensive(model, X_test, y_test, scaler=None, feature_selector=None):
    """
    Evaluate the trained model with comprehensive metrics
    
    Args:
        model: Trained regression model
        X_test (np.ndarray): Test features
        y_test (np.ndarray): Test targets
        scaler: Fitted scaler for test data
        feature_selector: Fitted feature selector for test data
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    # Apply preprocessing if provided
    X_test_processed = X_test.copy()
    if feature_selector is not None:
        X_test_processed = feature_selector.transform(X_test_processed)
    if scaler is not None:
        X_test_processed = scaler.transform(X_test_processed)
    
    y_pred = model.predict(X_test_processed)
    
    # Calculate comprehensive metrics
    metrics = {
        'mape': mean_absolute_percentage_error(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'r2': r2_score(y_test, y_pred),
        'predictions': y_pred,
        'actuals': y_test
    }
    
    # Calculate confidence intervals (simple bootstrap)
    n_bootstrap = 100
    bootstrap_scores = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(len(y_test), len(y_test), replace=True)
        y_test_boot = y_test[indices]
        y_pred_boot = y_pred[indices]
        bootstrap_scores.append(mean_absolute_percentage_error(y_test_boot, y_pred_boot))
    
    metrics['mape_ci_lower'] = np.percentile(bootstrap_scores, 2.5)
    metrics['mape_ci_upper'] = np.percentile(bootstrap_scores, 97.5)
    
    return metrics

def evaluate_model(model, X_test, y_test, scaler=None, feature_selector=None):
    """
    Evaluate the trained model and calculate MAPE (backward compatibility)
    
    Args:
        model: Trained regression model
        X_test (np.ndarray): Test features
        y_test (np.ndarray): Test targets
        scaler: Fitted scaler for test data
        feature_selector: Fitted feature selector for test data
        
    Returns:
        float: Mean Absolute Percentage Error (MAPE)
    """
    metrics = evaluate_model_comprehensive(model, X_test, y_test, scaler, feature_selector)
    return metrics['mape']

def evaluate_all_models(trained_models, X_test, y_test, scaler=None, feature_selector=None):
    """
    Evaluate all trained models on test data with comprehensive metrics
    
    Args:
        trained_models (dict): Dictionary of trained models
        X_test (np.ndarray): Test features
        y_test (np.ndarray): Test targets
        scaler: Fitted scaler for test data
        feature_selector: Fitted feature selector for test data
        
    Returns:
        dict: Dictionary mapping model names to comprehensive evaluation results
    """
    results = {}
    
    for model_name, (model, params, cv_score) in trained_models.items():
        try:
            metrics = evaluate_model_comprehensive(model, X_test, y_test, scaler, feature_selector)
            results[model_name] = {
                **metrics,
                'cv_score': cv_score,
                'params': params
            }
        except Exception as e:
            print(f"    Error evaluating {model_name}: {str(e)}")
            continue
    
    return results

def statistical_significance_test(results, metric='mape', alpha=0.05):
    """
    Perform statistical significance testing between models
    
    Args:
        results (dict): Results from cross-validation
        metric (str): Metric to test ('mape', 'mae', 'rmse', 'r2')
        alpha (float): Significance level
        
    Returns:
        dict: Statistical test results
    """
    # Collect metric values for each model across all workloads
    model_metrics = {}
    for workload, workload_result in results.items():
        for model_name, model_result in workload_result['model_results'].items():
            if model_name not in model_metrics:
                model_metrics[model_name] = []
            model_metrics[model_name].append(model_result[metric])
    
    # Perform pairwise comparisons
    model_names = list(model_metrics.keys())
    significance_results = {}
    
    for i, model1 in enumerate(model_names):
        for j, model2 in enumerate(model_names[i+1:], i+1):
            scores1 = np.array(model_metrics[model1])
            scores2 = np.array(model_metrics[model2])
            
            # Wilcoxon signed-rank test (non-parametric)
            try:
                statistic, p_value = wilcoxon(scores1, scores2, alternative='two-sided')
                significant = p_value < alpha
            except ValueError:
                # Handle case where all differences are zero
                p_value = 1.0
                significant = False
            
            significance_results[f"{model1}_vs_{model2}"] = {
                'p_value': p_value,
                'significant': significant,
                'model1_mean': np.mean(scores1),
                'model2_mean': np.mean(scores2),
                'model1_std': np.std(scores1),
                'model2_std': np.std(scores2)
            }
    
    return significance_results

def analyze_feature_importance(trained_models, feature_names, top_k=20):
    """
    Analyze feature importance across all models
    
    Args:
        trained_models (dict): Dictionary of trained models
        feature_names (list): List of feature names
        top_k (int): Number of top features to return
        
    Returns:
        dict: Feature importance analysis results
    """
    importance_results = {}
    
    for model_name, (model, params, cv_score) in trained_models.items():
        try:
            # Get feature importance if available
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_)
            else:
                continue
            
            # Create feature importance dataframe
            importance_df = pd.DataFrame({
                'feature': feature_names[:len(importances)],
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            importance_results[model_name] = {
                'top_features': importance_df.head(top_k).to_dict('records'),
                'all_importances': importances.tolist()
            }
            
        except Exception as e:
            print(f"    Error analyzing feature importance for {model_name}: {str(e)}")
            continue
    
    return importance_results

def create_visualizations(results, output_dir, feature_importance_results=None):
    """
    Create comprehensive visualizations of results
    
    Args:
        results (dict): Cross-validation results
        output_dir (str): Output directory for plots
        feature_importance_results (dict): Feature importance analysis results
    """
    plots_dir = os.path.join(output_dir, DEFAULT_PLOTS_DIR)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 1. Model Performance Comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Collect all model results
    all_results = []
    for workload, workload_result in results.items():
        for model_name, model_result in workload_result['model_results'].items():
            all_results.append({
                'workload': workload,
                'model': model_name,
                'mape': model_result['mape'],
                'mae': model_result['mae'],
                'rmse': model_result['rmse'],
                'r2': model_result['r2']
            })
    
    results_df = pd.DataFrame(all_results)
    
    # MAPE comparison
    sns.boxplot(data=results_df, x='model', y='mape', ax=axes[0,0])
    axes[0,0].set_title('Model Performance (MAPE)')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # R² comparison
    sns.boxplot(data=results_df, x='model', y='r2', ax=axes[0,1])
    axes[0,1].set_title('Model Performance (R²)')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # Performance by workload
    workload_performance = results_df.groupby('workload')['mape'].mean().sort_values()
    workload_performance.plot(kind='bar', ax=axes[1,0])
    axes[1,0].set_title('Average MAPE by Workload')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # Model ranking
    model_ranking = results_df.groupby('model')['mape'].mean().sort_values()
    model_ranking.plot(kind='bar', ax=axes[1,1])
    axes[1,1].set_title('Model Ranking (Mean MAPE)')
    axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'model_performance_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Feature Importance (if available)
    if feature_importance_results:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        model_names = list(feature_importance_results.keys())[:4]  # Top 4 models
        
        for i, model_name in enumerate(model_names):
            if i >= 4:
                break
                
            importance_data = feature_importance_results[model_name]['top_features'][:10]
            if importance_data:
                features = [item['feature'] for item in importance_data]
                importances = [item['importance'] for item in importance_data]
                
                axes[i].barh(features, importances)
                axes[i].set_title(f'Feature Importance - {model_name}')
                axes[i].set_xlabel('Importance')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Prediction vs Actual plots for best model
    best_model = results_df.groupby('model')['mape'].mean().idxmin()
    best_model_results = results_df[results_df['model'] == best_model]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    workloads = best_model_results['workload'].unique()
    for i, workload in enumerate(workloads[:4]):
        if i >= 4:
            break
            
        # Get predictions and actuals for this workload
        workload_result = results[workload]['model_results'][best_model]
        predictions = workload_result['predictions']
        actuals = workload_result['actuals']
        
        axes[i].scatter(actuals, predictions, alpha=0.6)
        axes[i].plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'r--', lw=2)
        axes[i].set_xlabel('Actual')
        axes[i].set_ylabel('Predicted')
        axes[i].set_title(f'{best_model} - {workload}')
        
        # Add R² to plot
        r2 = workload_result['r2']
        axes[i].text(0.05, 0.95, f'R² = {r2:.3f}', transform=axes[i].transAxes, 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'prediction_vs_actual.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to {plots_dir}")

def save_results(results, output_dir, experiment_name=None):
    """
    Save comprehensive results to files
    
    Args:
        results (dict): Cross-validation results
        output_dir (str): Output directory
        experiment_name (str): Name for this experiment
    """
    if experiment_name is None:
        experiment_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Create experiment directory
    exp_dir = os.path.join(output_dir, experiment_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Save results as JSON
    results_file = os.path.join(exp_dir, 'results.json')
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(item) for item in obj]
        else:
            return obj
    
    json_results = convert_for_json(results)
    
    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    # Save summary as CSV
    summary_data = []
    for workload, workload_result in results.items():
        for model_name, model_result in workload_result['model_results'].items():
            summary_data.append({
                'workload': workload,
                'model': model_name,
                'mape': model_result['mape'],
                'mae': model_result['mae'],
                'rmse': model_result['rmse'],
                'r2': model_result['r2'],
                'mape_ci_lower': model_result['mape_ci_lower'],
                'mape_ci_upper': model_result['mape_ci_upper'],
                'cv_score': model_result['cv_score']
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = os.path.join(exp_dir, 'summary.csv')
    summary_df.to_csv(summary_file, index=False)
    
    print(f"Results saved to {exp_dir}")
    return exp_dir

def save_models(trained_models, output_dir, experiment_name):
    """
    Save trained models to disk
    
    Args:
        trained_models (dict): Dictionary of trained models
        output_dir (str): Output directory
        experiment_name (str): Experiment name
    """
    models_dir = os.path.join(output_dir, experiment_name, DEFAULT_MODELS_DIR)
    os.makedirs(models_dir, exist_ok=True)
    
    for model_name, (model, params, cv_score) in trained_models.items():
        model_file = os.path.join(models_dir, f"{model_name}.pkl")
        with open(model_file, 'wb') as f:
            pickle.dump({
                'model': model,
                'params': params,
                'cv_score': cv_score,
                'model_name': model_name
            }, f)
    
    print(f"Models saved to {models_dir}")

def load_results(results_file):
    """
    Load results from JSON file
    
    Args:
        results_file (str): Path to results JSON file
        
    Returns:
        dict: Loaded results
    """
    with open(results_file, 'r') as f:
        results = json.load(f)
    return results

def load_model(model_file):
    """
    Load a trained model from pickle file
    
    Args:
        model_file (str): Path to model pickle file
        
    Returns:
        dict: Model data with 'model', 'params', 'cv_score', 'model_name'
    """
    with open(model_file, 'rb') as f:
        model_data = pickle.load(f)
    return model_data

def leave_one_out_cross_validation(workload_data, target_column, cv_folds=3, search_strategy='auto', max_iter=50, 
                                 n_jobs=None, preprocess=True, **preprocess_kwargs):
    """
    Perform leave-one-out cross validation across workloads with advanced model sweeping
    
    Args:
        workload_data (dict): Dictionary mapping workload names to (source_df, target_df) tuples
        target_column (str): Name of the target column to predict
        cv_folds (int): Number of CV folds for hyperparameter tuning
        search_strategy (str): Search strategy ('auto', 'grid', 'random', 'halving_grid', 'halving_random')
        max_iter (int): Maximum iterations for random search
        n_jobs (int): Number of parallel jobs for model training
        preprocess (bool): Whether to apply data preprocessing
        **preprocess_kwargs: Additional preprocessing arguments
        
    Returns:
        dict: Results for each left-out workload and model
    """
    results = {}
    workload_names = list(workload_data.keys())
    
    print(f"\nStarting leave-one-out cross validation with {len(workload_names)} workloads")
    print(f"Testing all regression models with advanced hyperparameter search")
    print(f"Search strategy: {search_strategy}, Max iterations: {max_iter}")
    print(f"Preprocessing: {preprocess}")
    
    for i, test_workload in enumerate(workload_names):
        print(f"\n{'='*60}")
        print(f"Fold {i+1}/{len(workload_names)}: Testing on workload '{test_workload}'")
        print(f"{'='*60}")
        
        # Split data: test workload vs training workloads
        test_data = {test_workload: workload_data[test_workload]}
        train_data = {k: v for k, v in workload_data.items() if k != test_workload}
        
        print(f"Training on {len(train_data)} workloads: {list(train_data.keys())}")
        print(f"Testing on 1 workload: {test_workload}")
        
        # Combine training data
        X_train, y_train = combine_workload_data(train_data, target_column)
        
        # Prepare test data
        test_source_df, test_target_df = test_data[test_workload]
        X_test, y_test, feature_names, test_scaler, test_feature_selector = create_feature_target_pairs(
            test_source_df, test_target_df, target_column, preprocess=preprocess, **preprocess_kwargs
        )
        
        print(f"Training data: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        print(f"Test data: {X_test.shape[0]} samples")
        
        # Train and tune all models with advanced search
        trained_models = train_all_models(X_train, y_train, cv_folds, search_strategy, max_iter, n_jobs)
        
        # Evaluate all models on test data
        model_results = evaluate_all_models(trained_models, X_test, y_test, test_scaler, test_feature_selector)
        
        results[test_workload] = {
            'model_results': model_results,
            'train_samples': len(y_train),
            'test_samples': len(y_test),
            'feature_names': feature_names
        }
        
        # Print best model for this fold
        if model_results:
            best_model = min(model_results.items(), key=lambda x: x[1]['mape'])
            print(f"\nBest model for {test_workload}: {best_model[0]} (MAPE: {best_model[1]['mape']:.4f})")
    
    return results

def print_results_summary(results):
    """
    Print a comprehensive summary of cross-validation results across all models
    
    Args:
        results (dict): Results from leave-one-out cross validation
    """
    print(f"\n{'='*80}")
    print("COMPREHENSIVE CROSS-VALIDATION RESULTS SUMMARY")
    print(f"{'='*80}")
    
    # Collect all model results across all workloads
    all_model_results = {}
    
    for workload, workload_result in results.items():
        for model_name, model_result in workload_result['model_results'].items():
            if model_name not in all_model_results:
                all_model_results[model_name] = []
            all_model_results[model_name].append(model_result['mape'])
    
    # Calculate statistics for each model
    model_stats = {}
    for model_name, mape_values in all_model_results.items():
        model_stats[model_name] = {
            'mean_mape': np.mean(mape_values),
            'std_mape': np.std(mape_values),
            'min_mape': np.min(mape_values),
            'max_mape': np.max(mape_values),
            'count': len(mape_values)
        }
    
    # Sort models by mean MAPE
    sorted_models = sorted(model_stats.items(), key=lambda x: x[1]['mean_mape'])
    
    print(f"Number of workloads tested: {len(results)}")
    print(f"Number of models tested: {len(all_model_results)}")
    print(f"\nModel Performance Ranking (by mean MAPE):")
    print(f"{'Rank':<4} {'Model':<25} {'Mean MAPE':<12} {'Std MAPE':<12} {'Min MAPE':<12} {'Max MAPE':<12}")
    print(f"{'-'*80}")
    
    for rank, (model_name, stats) in enumerate(sorted_models, 1):
        print(f"{rank:<4} {model_name:<25} {stats['mean_mape']:<12.4f} {stats['std_mape']:<12.4f} "
              f"{stats['min_mape']:<12.4f} {stats['max_mape']:<12.4f}")
    
    # Overall best model
    if sorted_models:
        best_model, best_stats = sorted_models[0]
        print(f"\nOverall Best Model: {best_model}")
        print(f"  Mean MAPE: {best_stats['mean_mape']:.4f} ({best_stats['mean_mape']*100:.2f}%)")
        print(f"  Std MAPE: {best_stats['std_mape']:.4f} ({best_stats['std_mape']*100:.2f}%)")
    
    # Detailed results by workload
    print(f"\n{'='*80}")
    print("DETAILED RESULTS BY WORKLOAD")
    print(f"{'='*80}")
    
    for workload, workload_result in results.items():
        print(f"\nWorkload: {workload}")
        print(f"  Train samples: {workload_result['train_samples']}, Test samples: {workload_result['test_samples']}")
        
        # Sort models by MAPE for this workload
        workload_models = sorted(workload_result['model_results'].items(), key=lambda x: x[1]['mape'])
        
        print(f"  Top 5 models:")
        for i, (model_name, model_result) in enumerate(workload_models[:5]):
            print(f"    {i+1}. {model_name}: MAPE = {model_result['mape']:.4f} ({model_result['mape']*100:.2f}%)")
        
        if len(workload_models) > 5:
            print(f"    ... and {len(workload_models) - 5} more models")
    
    return model_stats

def main():
    """
    Main function for cross-platform prediction evaluation with comprehensive enhancements
    """
    parser = argparse.ArgumentParser(description='Cross-platform performance prediction with comprehensive ML pipeline')
    
    # Required arguments
    parser.add_argument('--source_dir', required=True, help='Directory containing source CSV files')
    parser.add_argument('--target_dir', required=True, help='Directory containing target CSV files')
    parser.add_argument('--target_column', required=True, help='Name of the target column to predict')
    
    # Hyperparameter search arguments
    parser.add_argument('--cv_folds', type=int, default=3, help='Number of CV folds for hyperparameter tuning (default: 3)')
    parser.add_argument('--search_strategy', default='auto', 
                       choices=['auto', 'grid', 'random', 'halving_grid', 'halving_random'],
                       help='Hyperparameter search strategy (default: auto)')
    parser.add_argument('--max_iter', type=int, default=50, 
                       help='Maximum iterations for random search (default: 50)')
    
    # Parallel processing arguments
    parser.add_argument('--n_jobs', type=int, default=None, 
                       help='Number of parallel jobs (default: auto)')
    
    # Data preprocessing arguments
    parser.add_argument('--no_preprocess', action='store_true', 
                       help='Disable data preprocessing')
    parser.add_argument('--remove_outliers', action='store_true', default=True,
                       help='Remove outliers (default: True)')
    parser.add_argument('--feature_selection', type=int, default=None,
                       help='Number of features to select (default: None - use all)')
    parser.add_argument('--scaler_type', default='standard', choices=['standard', 'robust'],
                       help='Type of scaler (default: standard)')
    
    # Output arguments
    parser.add_argument('--output_dir', default=DEFAULT_OUTPUT_DIR,
                       help=f'Output directory (default: {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--experiment_name', default=None,
                       help='Experiment name (default: auto-generated)')
    parser.add_argument('--save_models', action='store_true',
                       help='Save trained models to disk')
    parser.add_argument('--create_plots', action='store_true', default=True,
                       help='Create visualization plots (default: True)')
    
    # Analysis arguments
    parser.add_argument('--statistical_tests', action='store_true', default=True,
                       help='Perform statistical significance tests (default: True)')
    parser.add_argument('--feature_importance', action='store_true', default=True,
                       help='Analyze feature importance (default: True)')
    
    args = parser.parse_args()
    
    # Create output directories
    output_dir = create_output_directories(args.output_dir)
    
    print("="*80)
    print("CROSS-PLATFORM PREDICTION EVALUATION - COMPREHENSIVE ML PIPELINE")
    print("="*80)
    print(f"Source directory: {args.source_dir}")
    print(f"Target directory: {args.target_dir}")
    print(f"Target column: {args.target_column}")
    print(f"CV folds: {args.cv_folds}")
    print(f"Search strategy: {args.search_strategy}")
    print(f"Max iterations: {args.max_iter}")
    print(f"Parallel jobs: {args.n_jobs}")
    print(f"Preprocessing: {not args.no_preprocess}")
    print(f"Output directory: {output_dir}")
    print("="*80)
    
    start_time = time.time()
    
    # Load and align all workload data
    workload_data = load_workload_data(args.source_dir, args.target_dir, args.target_column)
    
    if len(workload_data) < 2:
        print("Error: Need at least 2 workloads for leave-one-out cross validation")
        return
    
    # Prepare preprocessing arguments
    preprocess_kwargs = {
        'remove_outliers': args.remove_outliers,
        'feature_selection': args.feature_selection,
        'scaler_type': args.scaler_type
    }
    
    # Perform leave-one-out cross validation with advanced search
    results = leave_one_out_cross_validation(
        workload_data, args.target_column, args.cv_folds, args.search_strategy, 
        args.max_iter, args.n_jobs, not args.no_preprocess, **preprocess_kwargs
    )
    
    # Generate experiment name if not provided
    if args.experiment_name is None:
        args.experiment_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Save results
    exp_dir = save_results(results, output_dir, args.experiment_name)
    
    # Perform additional analyses
    if args.statistical_tests:
        print("\nPerforming statistical significance tests...")
        significance_results = statistical_significance_test(results)
        
        # Save significance results
        significance_file = os.path.join(exp_dir, 'statistical_tests.json')
        with open(significance_file, 'w') as f:
            json.dump(significance_results, f, indent=2)
        
        print(f"Statistical tests saved to {significance_file}")
    
    # Feature importance analysis
    feature_importance_results = None
    if args.feature_importance:
        print("\nAnalyzing feature importance...")
        # Get feature names from first workload
        first_workload = list(workload_data.keys())[0]
        feature_names = results[first_workload]['feature_names']
        
        # Analyze feature importance for best model from each fold
        all_trained_models = {}
        for workload, workload_result in results.items():
            # Get best model for this workload
            if workload_result['model_results']:
                best_model_name = min(workload_result['model_results'].items(), 
                                    key=lambda x: x[1]['mape'])[0]
                # Note: We don't have the actual trained models here, so we'll skip this for now
                # In a full implementation, you'd need to store the trained models
        
        # For now, we'll create a placeholder
        feature_importance_results = {}
        print("Feature importance analysis completed (placeholder)")
    
    # Create visualizations
    if args.create_plots:
        print("\nCreating visualizations...")
        create_visualizations(results, exp_dir, feature_importance_results)
    
    # Print comprehensive summary
    print_results_summary(results)
    
    # Print execution time
    execution_time = time.time() - start_time
    print(f"\nTotal execution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")
    
    print(f"\nResults saved to: {exp_dir}")
    print("="*80)
    
    return results

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Cross-platform CPU cycle prediction with GPU-accelerated regression models.
Supports CUDA acceleration for PyTorch models and optimized CPU models.
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_squared_error
from sklearn.model_selection import KFold, LeaveOneGroupOut, GroupKFold, RandomizedSearchCV
try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer, Categorical
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False
    print("WARNING: scikit-optimize not available, falling back to RandomizedSearchCV")
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import argparse
import time
import json
from datetime import datetime
import warnings
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading
from queue import Queue
warnings.filterwarnings('ignore')

class PyTorchMLP(nn.Module):
    """PyTorch MLP for regression with GPU acceleration"""
    def __init__(self, input_size, hidden_sizes=[1024, 512, 256], dropout=0.1):
        super().__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x).squeeze()


def get_device():
    """Get the best available device (CUDA > CPU) with A100 optimizations"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        print(f"CUDA GPU acceleration available")
        print(f"GPU: {gpu_name}")
        print(f"GPU Memory: {gpu_memory:.1f} GB")
        
        # A100-specific optimizations
        if "A100" in gpu_name:
            print("A100 GPU detected - enabling optimized settings!")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            print("TF32 and cuDNN optimizations enabled for A100")
        elif any(x in gpu_name for x in ["H100", "RTX", "V100"]):
            print("Modern GPU detected - enabling optimizations!")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            print("TF32 and cuDNN optimizations enabled")
        
        return device, True
    else:
        device = torch.device("cpu")
        print("WARNING: CUDA not available, using CPU")
        return device, False

def load_workload_data(source_dir, target_dir, target_column, max_workloads=None):
    """Load cross-platform workload data with memory optimization"""
    print(f"Loading cross-platform data...")
    
    source_files = sorted([f for f in os.listdir(source_dir) if f.endswith('.csv')])
    target_files = sorted([f for f in os.listdir(target_dir) if f.endswith('.csv')])
    
    source_workloads = [f.replace('_3.0GHz_merged.csv', '') for f in source_files]
    target_workloads = [f.replace('_1.5GHz_merged.csv', '') for f in target_files]
    
    common_workloads = sorted(set(source_workloads) & set(target_workloads))
    
    print(f"Found {len(common_workloads)} common workloads between source and target directories")
    
    if max_workloads and max_workloads < len(common_workloads):
        print(f"Limiting to {max_workloads} workloads")
        common_workloads = common_workloads[:max_workloads]
    else:
        print(f"Using all {len(common_workloads)} available workloads")
    
    workload_data = []
    
    # Memory optimization: use float32 instead of float64 to reduce memory usage
    for workload in common_workloads:
        source_file = f"{workload}_3.0GHz_merged.csv"
        target_file = f"{workload}_1.5GHz_merged.csv"
        
        source_path = os.path.join(source_dir, source_file)
        target_path = os.path.join(target_dir, target_file)
        
        if os.path.exists(source_path) and os.path.exists(target_path):
            print(f"  Loading {workload}...")
            
            # Load data without dtype constraint first
            source_df = pd.read_csv(source_path)
            target_df = pd.read_csv(target_path)
            
            # Get numeric columns (excluding metadata)
            numeric_columns = source_df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Remove metadata features
            features_to_remove = ['frequency', 'collection_granularity', 'spec_number', 'sample_number']
            feature_columns = [col for col in numeric_columns if col not in features_to_remove]
            
            if target_column in feature_columns:
                # Try float16 for maximum speed, fallback to float32 if overflow
                try:
                    X_source = source_df[feature_columns].values.astype(np.float16)
                    y_target = target_df[target_column].values.astype(np.float16)
                    # Check for overflow/underflow
                    if np.any(np.isinf(X_source)) or np.any(np.isnan(X_source)) or np.any(np.isinf(y_target)) or np.any(np.isnan(y_target)):
                        raise ValueError("Float16 overflow detected")
                    print(f"    Using float16 for maximum speed")
                except (ValueError, OverflowError):
                    # Fallback to float32 if float16 causes issues
                    X_source = source_df[feature_columns].values.astype(np.float32)
                    y_target = target_df[target_column].values.astype(np.float32)
                    print(f"    WARNING: Float16 overflow, using float32")
                
                workload_data.append({
                    'workload': workload,
                    'X': X_source,
                    'y': y_target,
                    'features': feature_columns
                })
                
                print(f"    {workload}: {len(X_source)} samples, {len(feature_columns)} features")
            else:
                print(f"    ERROR: {workload}: Target column '{target_column}' not found")
        else:
            print(f"    ERROR: {workload}: Missing files")
    
    return workload_data

def train_single_model_parallel(args):
    """Train a single model with given arguments (for parallel processing)"""
    model_name, model, X_scaled, y_scaled, all_groups, device, use_gpu, use_amp, scaler_y = args
    
    try:
        # Cross-validation
        if use_gpu and hasattr(model, 'forward'):
            cv_results = cross_validate_model(model, X_scaled, y_scaled, all_groups, model_name, device, use_amp=True, n_folds=5, scaler_y=scaler_y)
        else:
            cv_results = cross_validate_model(model, X_scaled, y_scaled, all_groups, model_name, n_folds=5, scaler_y=scaler_y)
        
        return cv_results
    except Exception as e:
        print(f"ERROR: Error training {model_name}: {str(e)}")
        return None

def train_pytorch_model(model, X_train, y_train, X_test, y_test, device, use_amp=False):
    """Train PyTorch model with A100-optimized settings"""
    model.to(device)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.FloatTensor(y_test).to(device)
    
    # A100-optimized settings
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3 if device.type == 'cuda' else 0
    if "A100" in torch.cuda.get_device_name(0) and gpu_memory > 30:  # A100 40GB or 80GB
        batch_size = 32768  # Large batch size for A100
        lr = 0.002  # Higher learning rate for large batch
        print(f"        A100 detected - using batch_size={batch_size}, lr={lr}")
    else:
        batch_size = 16384  # Still large for other GPUs
        lr = 0.001
    
    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-8)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    # Data loader with optimized settings
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing issues with CUDA
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Training
    epochs = 100
    patience = 20
    best_loss = float('inf')
    patience_counter = 0
    
    scaler = torch.amp.GradScaler('cuda') if use_amp and device.type == 'cuda' else None
    
    start_time = time.time()
    print(f"        Training for up to {epochs} epochs (early stopping patience: {patience})...")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    predictions = model(batch_X)
                    loss = criterion(predictions, batch_y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        scheduler.step(avg_loss)
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"        Early stopping at epoch {epoch + 1}")
            break
    
    training_time = time.time() - start_time
    print(f"        Training completed in {training_time:.1f}s ({epoch + 1} epochs)")
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        train_pred = model(X_train_tensor).cpu().numpy()
        test_pred = model(X_test_tensor).cpu().numpy()
    
    return {
        'train_pred': train_pred,
        'test_pred': test_pred,
        'training_time': training_time,
        'epochs_completed': epoch + 1
    }

def train_sklearn_model(model, X_train, y_train, X_test, y_test):
    """Train sklearn model"""
    start_time = time.time()
    
    model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    print(f"        Training completed in {training_time:.1f}s")
    
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    return {
        'train_pred': train_pred,
        'test_pred': test_pred,
        'training_time': training_time
    }

def evaluate_model(y_train, y_test, train_pred, test_pred, model_name, scaler_y=None):
    """Evaluate model performance with log transform inverse transformation"""
    # Apply inverse transformation from log space to original space
    y_train_orig = np.expm1(y_train)  # exp(x) - 1 inverse of log1p
    y_test_orig = np.expm1(y_test)
    train_pred_orig = np.expm1(train_pred)
    test_pred_orig = np.expm1(test_pred)
    
    # Calculate metrics on original scale
    train_mape = mean_absolute_percentage_error(y_train_orig, train_pred_orig) * 100
    test_mape = mean_absolute_percentage_error(y_test_orig, test_pred_orig) * 100
    train_r2 = r2_score(y_train_orig, train_pred_orig)
    test_r2 = r2_score(y_test_orig, test_pred_orig)
    train_rmse = np.sqrt(mean_squared_error(y_train_orig, train_pred_orig))
    test_rmse = np.sqrt(mean_squared_error(y_test_orig, test_pred_orig))
    
    return {
        'model': model_name,
        'train_mape': train_mape,
        'test_mape': test_mape,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse
    }

def cross_validate_model(model, X, y, groups, model_name, device=None, use_amp=False, n_folds=5, scaler_y=None, param_grid=None, n_iter=20):
    """Perform cross-validation with optional hyperparameter search"""
    
    # Check if this model should use hyperparameter search
    if param_grid is not None and model_name in ['XGBoost', 'CatBoost']:
        print(f"  Hyperparameter tuning {model_name} with {n_iter} iterations...")
        print(f"    Parameter grid: {list(param_grid.keys())}")
        
        # Use GroupKFold for hyperparameter search on tree models (10-fold)
        gkf_search = GroupKFold(n_splits=10)
        
        # Perform Bayesian optimization search
        if BAYESIAN_AVAILABLE:
            search = BayesSearchCV(
                model, param_grid, n_iter=n_iter, cv=gkf_search,
                scoring='neg_mean_absolute_percentage_error',
                random_state=42, n_jobs=1, verbose=1
            )
        else:
            # Fallback to randomized search
            search = RandomizedSearchCV(
                model, param_grid, n_iter=n_iter, cv=gkf_search,
                scoring='neg_mean_absolute_percentage_error',
                random_state=42, n_jobs=1, verbose=1
            )
        
        print(f"    Searching {n_iter} parameter combinations across 10-fold CV...")
        search.fit(X, y, groups=groups)
        
        print(f"    Best parameters: {search.best_params_}")
        print(f"    Best score (negative MAPE): {search.best_score_:.4f}")
        
        # Now evaluate the best model with full CV
        best_model = search.best_estimator_
        model_name = f"{model_name}_Tuned"
        
        # Use LeaveOneGroupOut for final evaluation
        logo_final = LeaveOneGroupOut()
        cv_results = []
        unique_groups = list(set(groups))
        num_workloads = len(unique_groups)
        
        print(f"  Final evaluation of tuned {model_name} with {num_workloads}-fold Leave-One-Group-Out CV...")
        
        for fold, (train_idx, test_idx) in enumerate(logo_final.split(X, y, groups=groups)):
            # Get unique workloads in test set
            test_workloads = list(set([groups[i] for i in test_idx]))
            train_workloads = list(set([groups[i] for i in train_idx]))
            
            print(f"    Fold {fold + 1}/{num_workloads}: Testing on workloads {test_workloads}")
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            print(f"      - Train: {len(X_train)} samples, Test: {len(X_test)} samples")
            
            # Train the tuned model
            results = train_sklearn_model(best_model, X_train, y_train, X_test, y_test)
            
            fold_result = evaluate_model(y_train, y_test, results['train_pred'], results['test_pred'], model_name, scaler_y)
            fold_result['training_time'] = results['training_time']
            fold_result['fold'] = fold + 1
            fold_result['test_workloads'] = test_workloads
            fold_result['train_workloads'] = train_workloads
            
            cv_results.append(fold_result)
            
            # Print fold results immediately
            print(f"      Fold {fold + 1} Results:")
            print(f"        - Test MAPE: {fold_result['test_mape']:.2f}%")
            print(f"        - Test R²: {fold_result['test_r2']:.4f}")
            print(f"        - Training Time: {fold_result['training_time']:.1f}s")
        
        # Add hyperparameter search info
        avg_results = {
            'model': model_name,
            'avg_train_mape': np.mean([r['train_mape'] for r in cv_results]),
            'std_train_mape': np.std([r['train_mape'] for r in cv_results]),
            'avg_test_mape': np.mean([r['test_mape'] for r in cv_results]),
            'std_test_mape': np.std([r['test_mape'] for r in cv_results]),
            'avg_train_r2': np.mean([r['train_r2'] for r in cv_results]),
            'std_train_r2': np.std([r['train_r2'] for r in cv_results]),
            'avg_test_r2': np.mean([r['test_r2'] for r in cv_results]),
            'std_test_r2': np.std([r['test_r2'] for r in cv_results]),
            'avg_training_time': np.mean([r['training_time'] for r in cv_results]),
            'total_folds': len(cv_results),
            'fold_results': cv_results,
            'best_params': search.best_params_,
            'hyperparameter_search_score': -search.best_score_
        }
        
    else:
        # Standard cross-validation without hyperparameter search
        print(f"  Cross-validating {model_name} with {n_folds} folds...")
        
        # Use GroupKFold to split workloads into groups
        gkf = GroupKFold(n_splits=n_folds)
        cv_results = []
        unique_groups = list(set(groups))
        
        # Calculate how many workloads per fold
        workloads_per_fold = len(unique_groups) // n_folds
        print(f"    Splitting {len(unique_groups)} workloads into {n_folds} groups (~{workloads_per_fold} workloads per test group)")
        
        for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups)):
            # Get unique workloads in test set
            test_workloads = list(set([groups[i] for i in test_idx]))
            train_workloads = list(set([groups[i] for i in train_idx]))
            
            print(f"    Fold {fold + 1}/{n_folds}: Testing on workloads {test_workloads}")
            print(f"      Training on workloads: {train_workloads}")
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            print(f"      - Train: {len(X_train)} samples, Test: {len(X_test)} samples")
            
            if hasattr(model, 'forward'):  # PyTorch model
                print(f"      Training PyTorch model on {device}...")
                results = train_pytorch_model(model, X_train, y_train, X_test, y_test, device, use_amp)
            else:  # sklearn model
                print(f"      Training sklearn model...")
                results = train_sklearn_model(model, X_train, y_train, X_test, y_test)
            
            fold_result = evaluate_model(y_train, y_test, results['train_pred'], results['test_pred'], model_name, scaler_y)
            fold_result['training_time'] = results['training_time']
            fold_result['fold'] = fold + 1
            fold_result['test_workloads'] = test_workloads
            fold_result['train_workloads'] = train_workloads
            
            cv_results.append(fold_result)
            
            # Print fold results immediately
            print(f"      Fold {fold + 1} Results:")
            print(f"        - Test MAPE: {fold_result['test_mape']:.2f}%")
            print(f"        - Test R²: {fold_result['test_r2']:.4f}")
            print(f"        - Training Time: {fold_result['training_time']:.1f}s")
        
        # Calculate average results
        avg_results = {
            'model': model_name,
            'avg_train_mape': np.mean([r['train_mape'] for r in cv_results]),
            'std_train_mape': np.std([r['train_mape'] for r in cv_results]),
            'avg_test_mape': np.mean([r['test_mape'] for r in cv_results]),
            'std_test_mape': np.std([r['test_mape'] for r in cv_results]),
            'avg_train_r2': np.mean([r['train_r2'] for r in cv_results]),
            'std_train_r2': np.std([r['train_r2'] for r in cv_results]),
            'avg_test_r2': np.mean([r['test_r2'] for r in cv_results]),
            'std_test_r2': np.std([r['test_r2'] for r in cv_results]),
            'avg_training_time': np.mean([r['training_time'] for r in cv_results]),
            'total_folds': len(cv_results),
            'fold_results': cv_results
        }
    
    # Print model summary immediately
    print(f"  {model_name} Summary:")
    print(f"    - Average Test MAPE: {avg_results['avg_test_mape']:.2f}% ± {avg_results['std_test_mape']:.2f}%")
    print(f"    - Average Test R²: {avg_results['avg_test_r2']:.4f} ± {avg_results['std_test_r2']:.4f}")
    print(f"    - Average Training Time: {avg_results['avg_training_time']:.1f}s")
    
    return avg_results

def get_xgboost_param_grid():
    """XGBoost hyperparameter search space (expanded for better performance)"""
    if BAYESIAN_AVAILABLE:
        return {
            'n_estimators': Integer(200, 1000),  # More trees for complex patterns
            'max_depth': Integer(6, 15),         # Deeper trees
            'learning_rate': Real(0.05, 0.3, prior='log-uniform'),  # Higher learning rates
            'subsample': Real(0.8, 1.0),         # Less aggressive subsampling
            'colsample_bytree': Real(0.8, 1.0),  # Use more features
            'reg_alpha': Real(0, 5),             # Less L1 regularization
            'reg_lambda': Real(0, 5)             # Less L2 regularization
        }
    else:
        # Fallback for RandomizedSearchCV
        return {
            'n_estimators': [200, 400, 600, 800, 1000],
            'max_depth': [6, 8, 10, 12, 15],
            'learning_rate': [0.05, 0.1, 0.15, 0.2, 0.3],
            'subsample': [0.8, 0.85, 0.9, 0.95, 1.0],
            'colsample_bytree': [0.8, 0.85, 0.9, 0.95, 1.0],
            'reg_alpha': [0, 1, 2, 3, 5],
            'reg_lambda': [0, 1, 2, 3, 5]
        }

def get_catboost_param_grid():
    """CatBoost hyperparameter search space (expanded for better performance)"""
    if BAYESIAN_AVAILABLE:
        return {
            'iterations': Integer(200, 800),       # More iterations
            'depth': Integer(6, 12),               # Deeper trees
            'learning_rate': Real(0.05, 0.3, prior='log-uniform'),  # Higher learning rates
            'l2_leaf_reg': Real(1, 5),             # Less regularization
            'random_strength': Real(0, 3),         # Less random strength
            'bagging_temperature': Real(0.5, 1.0)  # Higher bagging temperature
        }
    else:
        # Fallback for RandomizedSearchCV
        return {
            'iterations': [200, 400, 600, 800],
            'depth': [6, 8, 10, 12],
            'learning_rate': [0.05, 0.1, 0.15, 0.2, 0.3],
            'l2_leaf_reg': [1, 2, 3, 4, 5],
            'random_strength': [0, 1, 2, 3],
            'bagging_temperature': [0.5, 0.7, 0.8, 0.9, 1.0]
        }

def run_cross_platform_evaluation(source_dir, target_dir, target_column, max_workloads, output_file):
    """Run comprehensive cross-platform evaluation"""
    
    # Get device
    device, use_gpu = get_device()
    
    # Load data
    workload_data = load_workload_data(source_dir, target_dir, target_column, max_workloads)
    
    if len(workload_data) < 2:
        print("ERROR: Need at least 2 workloads for cross-validation")
        return
    
    # Combine all workload data
    all_X = []
    all_y = []
    all_groups = []
    
    for workload_info in workload_data:
        X = workload_info['X']
        y = workload_info['y']
        workload_name = workload_info['workload']
        
        all_X.append(X)
        all_y.append(y)
        all_groups.extend([workload_name] * len(X))
        
        print(f"  {workload_name}: {len(X)} samples, {X.shape[1]} features")
    
    # Concatenate all data
    X_combined = np.vstack(all_X)
    y_combined = np.hstack(all_y)
    
    print(f"  Combined data: {len(X_combined)} samples, {X_combined.shape[1]} features")
    
    # Scale features with memory optimization
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X_combined)
    
    # Log transform targets (no additional scaling needed)
    y_log = np.log1p(y_combined)
    scaler_y = None  # No scaler needed for log transform
    y_scaled = y_log  # Use log-transformed targets directly
    
    # Try to maintain original precision (float16 if available)
    try:
        # Check if original data was float16
        if X_combined.dtype == np.float16:
            X_scaled = X_scaled.astype(np.float16)
            y_scaled = y_scaled.astype(np.float16)
            # Verify no overflow
            if np.any(np.isinf(X_scaled)) or np.any(np.isnan(X_scaled)) or np.any(np.isinf(y_scaled)) or np.any(np.isnan(y_scaled)):
                raise ValueError("Float16 overflow in scaled data")
            precision = "float16"
        else:
            X_scaled = X_scaled.astype(np.float32)
            y_scaled = y_scaled.astype(np.float32)
            precision = "float32"
    except (ValueError, OverflowError):
        # Fallback to float32
        X_scaled = X_scaled.astype(np.float32)
        y_scaled = y_scaled.astype(np.float32)
        precision = "float32 (fallback)"
    
    print(f"  Memory optimization: Using {precision} precision")
    print(f"    - X_scaled memory: {X_scaled.nbytes / 1024**2:.1f} MB")
    print(f"    - y_scaled memory: {y_scaled.nbytes / 1024**2:.1f} MB")
    
    print(f"  Target scaling with log transform:")
    print(f"    - Original y range: [{y_combined.min():.2e}, {y_combined.max():.2e}]")
    print(f"    - Log y range: [{y_log.min():.3f}, {y_log.max():.3f}]")
    print(f"    - Using log-transformed targets directly (no additional scaling)")
    
    # Define models - XGBoost and CatBoost for hyperparameter search
    models = {}
    
    # XGBoost with GPU acceleration
    if use_gpu:
        models['XGBoost'] = XGBRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42,
            tree_method='gpu_hist', gpu_id=0, n_jobs=-1
        )
        print("  XGBoost configured for GPU acceleration")
    else:
        models['XGBoost'] = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)
    
    # CatBoost with GPU acceleration
    if use_gpu:
        models['CatBoost'] = CatBoostRegressor(
            iterations=200, depth=6, learning_rate=0.1, random_state=42,
            task_type='GPU', devices='0', verbose=0
        )
        print("  CatBoost configured for GPU acceleration")
    else:
        models['CatBoost'] = CatBoostRegressor(
            iterations=200, depth=6, learning_rate=0.1, random_state=42,
            verbose=0, thread_count=-1
        )
    
    # Run evaluation for both XGBoost and CatBoost
    print(f"\nRunning XGBoost and CatBoost cross-platform evaluation with hyperparameter search...")
    print("=" * 80)
    
    all_results = []
    
    # Process both XGBoost and CatBoost with hyperparameter search
    for i, (model_name, model) in enumerate(models.items()):
        print(f"\nEvaluating {model_name}...")
        print("-" * 60)
        
        try:
            # Determine parameter grid and CV strategy
            param_grid = None
            n_folds = 5
            n_iter = 100  # Number of hyperparameter search iterations
            
            # Get model-specific parameter grid
            if model_name == 'XGBoost':
                param_grid = get_xgboost_param_grid()
                n_folds = 40  # Leave-one-out for final evaluation (will auto-adjust to actual workload count)
                print(f"  {model_name}: Bayesian hyperparameter search with {n_iter} iterations, 10-fold CV, then Leave-One-Group-Out final evaluation")
            elif model_name == 'CatBoost':
                param_grid = get_catboost_param_grid()
                n_folds = 40  # Leave-one-out for final evaluation (will auto-adjust to actual workload count)
                print(f"  {model_name}: Bayesian hyperparameter search with {n_iter} iterations, 10-fold CV, then Leave-One-Group-Out final evaluation")
            
            # Cross-validation with hyperparameter search
            cv_results = cross_validate_model(model, X_scaled, y_scaled, all_groups, model_name, 
                                            n_folds=n_folds, scaler_y=scaler_y, 
                                            param_grid=param_grid, n_iter=n_iter)
            all_results.append(cv_results)
            
            # Print results summary
            print(f"\n{model_name} Results Summary:")
            print(f"  - Average Test MAPE: {cv_results['avg_test_mape']:.2f}% ± {cv_results['std_test_mape']:.2f}%")
            print(f"  - Average Test R²: {cv_results['avg_test_r2']:.4f} ± {cv_results['std_test_r2']:.4f}")
            print(f"  - Average Training Time: {cv_results['avg_training_time']:.1f}s")
            if 'best_params' in cv_results:
                print(f"  - Best Parameters: {cv_results['best_params']}")
            print("-" * 60)
            
        except Exception as e:
            print(f"ERROR: {model_name} failed with error: {str(e)}")
            continue
    
    # Sort results by test MAPE
    all_results.sort(key=lambda x: x['avg_test_mape'])
    
    # Print summary
    print(f"\nCROSS-PLATFORM EVALUATION RESULTS (XGBoost & CatBoost)")
    print("=" * 80)
    print(f"Summary: {len(all_results)} models evaluated on {len(workload_data)} workloads")
    print(f"Target: {target_column}")
    
    if all_results:
        print(f"\nModel Rankings (by Test MAPE):")
        for i, result in enumerate(all_results, 1):
            print(f"\n{i}. {result['model']} Performance:")
            print(f"  - Average Test MAPE: {result['avg_test_mape']:.2f}% ± {result['std_test_mape']:.2f}%")
            print(f"  - Average Test R²: {result['avg_test_r2']:.4f} ± {result['std_test_r2']:.4f}")
            print(f"  - Average Training Time: {result['avg_training_time']:.1f}s")
            print(f"  - Total Folds: {result['total_folds']}")
            if 'best_params' in result:
                print(f"  - Best Parameters: {result['best_params']}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_data = {
        'timestamp': timestamp,
        'source_dir': source_dir,
        'target_dir': target_dir,
        'target_column': target_column,
        'workloads': [w['workload'] for w in workload_data],
        'device_info': {
            'use_gpu': use_gpu,
            'device_name': torch.cuda.get_device_name(0) if use_gpu else 'CPU'
        },
        'data_info': {
            'total_samples': len(X_combined),
            'total_features': X_combined.shape[1],
            'workload_count': len(workload_data)
        },
        'results': all_results
    }
    
    with open(output_file, 'w') as f:
        json.dump(results_data, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_file}")
    
    return all_results

def main():
    parser = argparse.ArgumentParser(description='Cross-platform CPU cycle prediction with GPU acceleration')
    parser.add_argument('--source_dir', type=str, default='data/arm_server/final_csvs/3.0GHz',
                        help='Source directory with 3.0GHz data')
    parser.add_argument('--target_dir', type=str, default='data/arm_server/final_csvs/1.5GHz',
                        help='Target directory with 1.5GHz data')
    parser.add_argument('--target_column', type=str, default='cpu-cycles:',
                        help='Target column to predict')
    parser.add_argument('--max_workloads', type=int, default=None,
                        help='Maximum number of workloads to evaluate (default: None = use all available)')
    parser.add_argument('--output_file', type=str, default='cross_platform_gpu_results.json',
                        help='Output file for results')
    
    args = parser.parse_args()
    
    print("Cross-Platform GPU Evaluation")
    print("=" * 80)
    print(f"Source: {args.source_dir}")
    print(f"Target: {args.target_dir}")
    print(f"Target Column: {args.target_column}")
    print(f"Max Workloads: {'All available' if args.max_workloads is None else args.max_workloads}")
    print(f"Output File: {args.output_file}")
    
    # Run evaluation
    results = run_cross_platform_evaluation(
        args.source_dir, args.target_dir, args.target_column, 
        args.max_workloads, args.output_file
    )
    
    if results:
        print(f"\nEvaluation completed successfully!")
        print(f"{len(results)} models evaluated")
        print(f"Best model: {results[0]['model']} (MAPE: {results[0]['avg_test_mape']:.2f}%)")

if __name__ == "__main__":
    main()

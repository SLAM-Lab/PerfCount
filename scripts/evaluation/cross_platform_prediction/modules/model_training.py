"""
Model training module for cross-platform prediction evaluation.

This module handles model training, hyperparameter tuning, and parallel
processing.
"""

import numpy as np
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV, HalvingRandomSearchCV
)
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from .model_config import get_model_configurations


def _reduce_search_space(models):
    """Reduce hyperparameter search space for fast mode"""
    reduced_models = {}
    for model_name, (model, param_grid) in models.items():
        reduced_grid = {}
        for param, values in param_grid.items():
            if len(values) > 3:
                # Take first, middle, and last values
                if len(values) == 4:
                    reduced_grid[param] = [values[0], values[2], values[3]]
                else:
                    step = len(values) // 3
                    reduced_grid[param] = [values[0], values[step], values[-1]]
            else:
                reduced_grid[param] = values
        reduced_models[model_name] = (model, reduced_grid)
    return reduced_models


def _select_diverse_models(models, max_models):
    """Select diverse models covering different algorithm types"""
    # Define model categories
    linear_models = [k for k in models.keys() if any(x in k.lower() for x in 
                   ['linear', 'ridge', 'lasso', 'elastic', 'bayesian', 'huber', 'sgd', 'passive'])]
    tree_models = [k for k in models.keys() if any(x in k.lower() for x in 
                  ['tree', 'forest', 'extra'])]
    ensemble_models = [k for k in models.keys() if any(x in k.lower() for x in 
                      ['gradient', 'ada', 'bagging', 'voting', 'stacking', 'xgb', 'catboost', 'lgbm'])]
    svm_models = [k for k in models.keys() if 'svr' in k.lower()]
    knn_models = [k for k in models.keys() if 'kneighbors' in k.lower()]
    neural_models = [k for k in models.keys() if any(x in k.lower() for x in 
                    ['mlp', 'pytorch', 'gaussian'])]
    other_models = [k for k in models.keys() if not any(x in k.lower() for x in 
                   ['linear', 'tree', 'forest', 'extra', 'gradient', 'ada', 'bagging', 
                    'voting', 'stacking', 'xgb', 'catboost', 'lgbm', 'svr', 'kneighbors', 
                    'mlp', 'pytorch', 'gaussian'])]
    
    # Select models from each category
    selected = []
    categories = [linear_models, tree_models, ensemble_models, svm_models, 
                 knn_models, neural_models, other_models]
    
    for category in categories:
        if category and len(selected) < max_models:
            selected.append(category[0])
    
    # Fill remaining slots with best models
    remaining_models = [k for k in models.keys() if k not in selected]
    while len(selected) < max_models and remaining_models:
        selected.append(remaining_models.pop(0))
    
    return selected


def train_and_tune_model(model_name, model, param_grid, X_train, y_train,
                         cv_folds=3, search_strategy='auto', max_iter=50):
    """
    Train and hyperparameter tune a regression model with advanced search
    strategies
    
    Args:
        model_name (str): Name of the model
        model: Model instance
        param_grid (dict): Hyperparameter grid
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training targets
        cv_folds (int): Number of CV folds for hyperparameter tuning
        search_strategy (str): Search strategy ('auto', 'grid', 'random',
            'halving_grid', 'halving_random')
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
        args: Tuple of (model_name, model, param_grid, X_train, y_train,
            cv_folds, search_strategy, max_iter)
        
    Returns:
        tuple: (model_name, best_model, best_params, best_score) or
            (model_name, None, None, None) if failed
    """
    (model_name, model, param_grid, X_train, y_train, cv_folds,
     search_strategy, max_iter) = args
    
    try:
        best_model, best_params, best_score = train_and_tune_model(
            model_name, model, param_grid, X_train, y_train, cv_folds,
            search_strategy, max_iter
        )
        return model_name, best_model, best_params, best_score
    except Exception as e:
        print(f"    Error training {model_name}: {str(e)}")
        return model_name, None, None, None


def train_all_models(X_train, y_train, cv_folds=3, search_strategy='auto',
                     max_iter=50, n_jobs=None, max_models=None, fast_mode=False):
    """
    Train and tune all regression models with advanced hyperparameter search and parallel processing
    
    Args:
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training targets
        cv_folds (int): Number of CV folds for hyperparameter tuning
        search_strategy (str): Search strategy ('auto', 'grid', 'random',
            'halving_grid', 'halving_random')
        max_iter (int): Maximum iterations for random search
        n_jobs (int): Number of parallel jobs (None for auto)
        max_models (int): Maximum number of models to train (None for all)
        fast_mode (bool): Reduce hyperparameter search space for speed
        
    Returns:
        dict: Dictionary mapping model names to (model, params, cv_score) tuples
    """
    print(f"Training and tuning all models on {X_train.shape[0]} samples with {X_train.shape[1]} features")
    print(f"Search strategy: {search_strategy}, Max iterations: {max_iter}")
    
    models = get_model_configurations()
    
    # Apply fast mode if requested
    if fast_mode:
        print("Fast mode enabled: reducing hyperparameter search space")
        models = _reduce_search_space(models)
    
    # Limit number of models if requested
    if max_models is not None and len(models) > max_models:
        print(f"Limiting to {max_models} models (from {len(models)} available)")
        # Select diverse models: linear, tree-based, ensemble, neural network
        selected_models = _select_diverse_models(models, max_models)
        models = {k: v for k, v in models.items() if k in selected_models}
    
    if n_jobs is None:
        # Use all available cores for maximum parallelism
        n_jobs = cpu_count()
    
    print(f"Training {len(models)} models using {n_jobs} parallel jobs (all available cores)")
    
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

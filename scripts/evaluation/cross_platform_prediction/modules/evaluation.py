"""
Evaluation module for cross-platform prediction evaluation.

This module handles model evaluation, metrics calculation, and statistical
analysis.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_percentage_error, mean_absolute_error, mean_squared_error,
    r2_score
)
from scipy.stats import wilcoxon


def evaluate_model_comprehensive(model, X_test, y_test, scaler=None,
                                 feature_selector=None):
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
        bootstrap_scores.append(
            mean_absolute_percentage_error(y_test_boot, y_pred_boot)
        )
    
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
    metrics = evaluate_model_comprehensive(
        model, X_test, y_test, scaler, feature_selector
    )
    return metrics['mape']


def evaluate_all_models(trained_models, X_test, y_test, scaler=None,
                        feature_selector=None):
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
            metrics = evaluate_model_comprehensive(
        model, X_test, y_test, scaler, feature_selector
    )
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
        for model_name, model_result in workload_result[
                'model_results'
        ].items():
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
                statistic, p_value = wilcoxon(
                    scores1, scores2, alternative='two-sided'
                )
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
            print(f"    Error analyzing feature importance for {model_name}: "
                  f"{str(e)}")
            continue
    
    return importance_results


def print_results_summary(results):
    """
    Print a comprehensive summary of cross-validation results across all models
    
    Args:
        results (dict): Results from leave-one-out cross validation
        
    Returns:
        dict: Model statistics
    """
    print(f"\n{'='*80}")
    print("COMPREHENSIVE CROSS-VALIDATION RESULTS SUMMARY")
    print(f"{'='*80}")
    
    # Collect all model results across all workloads
    all_model_results = {}
    
    for workload, workload_result in results.items():
        for model_name, model_result in workload_result[
                'model_results'
        ].items():
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

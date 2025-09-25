"""
Utility module for cross-platform prediction evaluation.

This module contains utility functions for file I/O, directory management,
and data serialization.
"""

import os
import json
import pickle
import numpy as np
from datetime import datetime


# Configuration
DEFAULT_OUTPUT_DIR = "results"
DEFAULT_MODELS_DIR = "models"
DEFAULT_PLOTS_DIR = "plots"


def create_output_directories(output_dir=DEFAULT_OUTPUT_DIR):
    """
    Create output directories for results, models, and plots
    
    Args:
        output_dir (str): Base output directory
        
    Returns:
        str: Path to created output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, DEFAULT_MODELS_DIR), exist_ok=True)
    os.makedirs(os.path.join(output_dir, DEFAULT_PLOTS_DIR), exist_ok=True)
    return output_dir


def save_results(results, output_dir, experiment_name=None):
    """
    Save comprehensive results to files
    
    Args:
        results (dict): Cross-validation results
        output_dir (str): Output directory
        experiment_name (str): Name for this experiment
        
    Returns:
        str: Path to experiment directory
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
    
    import pandas as pd
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
        
    Returns:
        str: Path to models directory
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
    return models_dir


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

"""
Cross-platform prediction evaluation script.

This script performs comprehensive cross-platform performance prediction
using leave-one-out cross-validation with advanced hyperparameter search
across multiple machine learning models.
"""

import argparse
import time
import json
import os
from datetime import datetime

# Import our modular components
from modules import (
    data_processing,
    model_training, 
    evaluation,
    visualization,
    utils
)


def leave_one_out_cross_validation(workload_data, target_column, cv_folds=5,
                                   search_strategy='auto', max_iter=50,
                                   n_jobs=None, preprocess=True,
                                   **preprocess_kwargs):
    """
    Perform leave-one-out cross validation across workloads with advanced
    model sweeping
    
    Args:
        workload_data (dict): Dictionary mapping workload names to
            (source_df, target_df) tuples
        target_column (str): Name of the target column to predict
        cv_folds (int): Number of CV folds for hyperparameter tuning
        search_strategy (str): Search strategy ('auto', 'grid', 'random',
            'halving_grid', 'halving_random')
        max_iter (int): Maximum iterations for random search
        n_jobs (int): Number of parallel jobs for model training
        preprocess (bool): Whether to apply data preprocessing
        **preprocess_kwargs: Additional preprocessing arguments
        
    Returns:
        dict: Results for each left-out workload and model
    """
    results = {}
    workload_names = list(workload_data.keys())
    
    print(f"\nStarting leave-one-out cross validation with "
          f"{len(workload_names)} workloads")
    print(f"Testing all regression models with advanced hyperparameter search")
    print(f"Search strategy: {search_strategy}, Max iterations: {max_iter}")
    print(f"Preprocessing: {preprocess}")
    
    for i, test_workload in enumerate(workload_names):
        print(f"\n{'='*60}")
        print(f"Fold {i+1}/{len(workload_names)}: Testing on workload "
              f"'{test_workload}'")
        print(f"{'='*60}")
        
        # Split data: test workload vs training workloads
        test_data = {test_workload: workload_data[test_workload]}
        train_data = {k: v for k, v in workload_data.items()
                      if k != test_workload}
        
        print(f"Training on {len(train_data)} workloads: "
              f"{list(train_data.keys())}")
        print(f"Testing on 1 workload: {test_workload}")
        
        # Combine training data
        X_train, y_train = data_processing.combine_workload_data(
            train_data, target_column
        )
        
        # Prepare test data
        test_source_df, test_target_df = test_data[test_workload]
        X_test, y_test, feature_names, test_scaler, test_feature_selector = (
            data_processing.create_feature_target_pairs(
                test_source_df, test_target_df, target_column,
                preprocess=preprocess, **preprocess_kwargs
            )
        )
        
        print(f"Training data: {X_train.shape[0]} samples, "
              f"{X_train.shape[1]} features")
        print(f"Test data: {X_test.shape[0]} samples")
        
        # Train and tune all models with advanced search
        trained_models = model_training.train_all_models(
            X_train, y_train, cv_folds, search_strategy, max_iter, n_jobs,
            args.max_models, args.fast_mode
        )
        
        # Evaluate all models on test data
        model_results = evaluation.evaluate_all_models(trained_models, X_test, y_test, test_scaler, test_feature_selector)
        
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
    parser.add_argument('--cv_folds', type=int, default=5, help='Number of CV folds for hyperparameter tuning (default: 5)')
    parser.add_argument('--search_strategy', default='auto', 
                       choices=['auto', 'grid', 'random', 'halving_grid', 'halving_random'],
                       help='Hyperparameter search strategy (default: auto)')
    parser.add_argument('--max_iter', type=int, default=50, 
                       help='Maximum iterations for random search (default: 50)')
    
    # Parallel processing arguments
    parser.add_argument('--n_jobs', type=int, default=None, 
                       help='Number of parallel jobs (default: all cores)')
    parser.add_argument('--use_gpu', action='store_true', default=True,
                       help='Use GPU acceleration when available (default: True)')
    parser.add_argument('--no_gpu', action='store_true',
                       help='Disable GPU acceleration (force CPU only)')
    parser.add_argument('--max_models', type=int, default=None,
                       help='Maximum number of models to train (for testing)')
    parser.add_argument('--fast_mode', action='store_true',
                       help='Fast mode: reduce hyperparameter search space')
    
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
    parser.add_argument('--output_dir', default=utils.DEFAULT_OUTPUT_DIR,
                       help=f'Output directory (default: {utils.DEFAULT_OUTPUT_DIR})')
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
    output_dir = utils.create_output_directories(args.output_dir)
    
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
    print(f"GPU acceleration: {args.use_gpu and not args.no_gpu}")
    print(f"Preprocessing: {not args.no_preprocess}")
    print(f"Output directory: {output_dir}")
    print("="*80)
    
    start_time = time.time()
    
    # Load and align all workload data
    workload_data = data_processing.load_workload_data(args.source_dir, args.target_dir, args.target_column)
    
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
    exp_dir = utils.save_results(results, output_dir, args.experiment_name)
    
    # Perform additional analyses
    if args.statistical_tests:
        print("\nPerforming statistical significance tests...")
        significance_results = evaluation.statistical_significance_test(results)
        
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
        
        # For now, we'll create a placeholder since we don't store trained models
        feature_importance_results = {}
        print("Feature importance analysis completed (placeholder)")
    
    # Create visualizations
    if args.create_plots:
        print("\nCreating visualizations...")
        visualization.create_visualizations(results, exp_dir, feature_importance_results)
    
    # Print comprehensive summary
    evaluation.print_results_summary(results)
    
    # Print execution time
    execution_time = time.time() - start_time
    print(f"\nTotal execution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")
    
    print(f"\nResults saved to: {exp_dir}")
    print("="*80)
    
    return results


if __name__ == "__main__":
    main()

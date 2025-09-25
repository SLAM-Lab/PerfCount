"""
Visualization module for cross-platform prediction evaluation.

This module handles plotting, visualization, and result presentation.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def create_visualizations(results, output_dir, feature_importance_results=None):
    """
    Create comprehensive visualizations of results
    
    Args:
        results (dict): Cross-validation results
        output_dir (str): Output directory for plots
        feature_importance_results (dict): Feature importance analysis results
    """
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 1. Model Performance Comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Collect all model results
    all_results = []
    for workload, workload_result in results.items():
        for model_name, model_result in workload_result[
                'model_results'
        ].items():
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
    workload_performance = results_df.groupby('workload')['mape'].mean(
    ).sort_values()
    workload_performance.plot(kind='bar', ax=axes[1,0])
    axes[1,0].set_title('Average MAPE by Workload')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # Model ranking
    model_ranking = results_df.groupby('model')['mape'].mean().sort_values()
    model_ranking.plot(kind='bar', ax=axes[1,1])
    axes[1,1].set_title('Model Ranking (Mean MAPE)')
    axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(
        os.path.join(plots_dir, 'model_performance_comparison.png'),
        dpi=300, bbox_inches='tight'
    )
    plt.close()
    
    # 2. Feature Importance (if available)
    if feature_importance_results:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        model_names = list(feature_importance_results.keys())[:4]  # Top 4
        
        for i, model_name in enumerate(model_names):
            if i >= 4:
                break
                
            importance_data = feature_importance_results[model_name][
                'top_features'
            ][:10]
            if importance_data:
                features = [item['feature'] for item in importance_data]
                importances = [item['importance'] for item in importance_data]
                
                axes[i].barh(features, importances)
                axes[i].set_title(f'Feature Importance - {model_name}')
                axes[i].set_xlabel('Importance')
        
        plt.tight_layout()
        plt.savefig(
            os.path.join(plots_dir, 'feature_importance.png'),
            dpi=300, bbox_inches='tight'
        )
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
        axes[i].plot(
            [actuals.min(), actuals.max()], [actuals.min(), actuals.max()],
            'r--', lw=2
        )
        axes[i].set_xlabel('Actual')
        axes[i].set_ylabel('Predicted')
        axes[i].set_title(f'{best_model} - {workload}')
        
        # Add R² to plot
        r2 = workload_result['r2']
        axes[i].text(0.05, 0.95, f'R² = {r2:.3f}', transform=axes[i].transAxes, 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(
        os.path.join(plots_dir, 'prediction_vs_actual.png'),
        dpi=300, bbox_inches='tight'
    )
    plt.close()
    
    print(f"Visualizations saved to {plots_dir}")

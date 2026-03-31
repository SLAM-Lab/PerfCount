#!/usr/bin/env python3
"""
Process workload forecasting log files and extract performance metrics.

This script reads log files from forecasting experiments and extracts metrics
like MAPE, RMSE, R2, etc., then generates summary statistics grouped by model.

Usage:
    python process_log_results.py --input counter_forecasting_logs/BUS_CYCLES \
                                   --output counter_forecasting_logs/BUS_CYCLES

    ./process_log_results.py --input logs/ --output results/
"""

import os
import re
import ast
import glob
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict


def parse_log_file(log_path):
    """
    Parse a single log file and extract metrics.
    
    Returns:
        dict: Dictionary with extracted metrics, or None if parsing failed
    """
    try:
        with open(log_path, 'r') as f:
            content = f.read()
        
        # Find the Metrics line
        metrics_match = re.search(r"Metrics:\s*(\{[^}]+\})", content)
        
        if not metrics_match:
            return None
        
        # Parse the metrics dictionary
        metrics_str = metrics_match.group(1)
        metrics = ast.literal_eval(metrics_str)
        
        # Extract model and workload info from filename
        filename = os.path.basename(log_path)
        
        # Filename format: {workload}_{model}_{counter}_t{timesteps}_h{horizon}.log
        # Need to extract model type (lstm, transformer, dt, mlp, stacked_lstm)
        
        # Common model types
        model_types = ['stacked_lstm', 'lstm', 'transformer', 'mlp', 'dt']
        model = None
        
        for model_type in model_types:
            if f'_{model_type}_' in filename:
                model = model_type
                break
        
        if not model:
            # Try to extract from filename more carefully
            parts = filename.replace('.log', '').split('_')
            for part in parts:
                if part in model_types:
                    model = part
                    break
        
        # Extract workload (everything before model)
        if model:
            workload = filename.split(f'_{model}_')[0]
        else:
            workload = 'unknown'
            model = 'unknown'
        
        # Add metadata to metrics
        result = {
            'filename': filename,
            'workload': workload,
            'model': model,
            **metrics
        }
        
        return result
        
    except Exception as e:
        print(f"Error parsing {log_path}: {e}")
        return None


def process_directory(input_dir):
    """
    Process all log files in a directory.
    
    Args:
        input_dir: Directory containing log files
        
    Returns:
        pd.DataFrame: DataFrame with all extracted metrics
    """
    log_files = glob.glob(os.path.join(input_dir, '*.log'))
    
    print(f"Found {len(log_files)} log files in {input_dir}")
    
    results = []
    failed = []
    
    for log_file in log_files:
        metrics = parse_log_file(log_file)
        if metrics:
            results.append(metrics)
        else:
            failed.append(os.path.basename(log_file))
    
    print(f"Successfully parsed: {len(results)}")
    print(f"Failed to parse: {len(failed)}")
    
    if failed and len(failed) <= 10:
        print(f"Failed files: {', '.join(failed)}")
    elif failed:
        print(f"First 10 failed files: {', '.join(failed[:10])}")
    
    if not results:
        return None
    
    df = pd.DataFrame(results)
    return df


def generate_summary(df, output_dir):
    """
    Generate summary statistics grouped by model.
    
    Args:
        df: DataFrame with metrics
        output_dir: Directory to save summary files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Metrics to summarize
    metric_cols = ['mape', 'rmse', 'r2_score', 'mpe', 'exp_var']
    available_metrics = [col for col in metric_cols if col in df.columns]
    
    print("\n" + "="*80)
    print("SUMMARY BY MODEL")
    print("="*80)
    
    # Group by model
    summary_data = []
    
    for model in sorted(df['model'].unique()):
        model_df = df[df['model'] == model]
        
        print(f"\n{model.upper()}")
        print("-" * 80)
        print(f"Number of workloads: {len(model_df)}")
        
        row = {'model': model, 'count': len(model_df)}
        
        for metric in available_metrics:
            if metric in model_df.columns:
                values = model_df[metric].dropna()
                if len(values) > 0:
                    mean_val = values.mean()
                    std_val = values.std()
                    median_val = values.median()
                    min_val = values.min()
                    max_val = values.max()
                    
                    row[f'{metric}_mean'] = mean_val
                    row[f'{metric}_std'] = std_val
                    row[f'{metric}_median'] = median_val
                    row[f'{metric}_min'] = min_val
                    row[f'{metric}_max'] = max_val
                    
                    print(f"  {metric.upper():15s}: {mean_val:8.4f} ± {std_val:7.4f}  "
                          f"(median: {median_val:8.4f}, min: {min_val:8.4f}, max: {max_val:8.4f})")
        
        summary_data.append(row)
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # Sort by MAPE mean (best performing first)
    if 'mape_mean' in summary_df.columns:
        summary_df = summary_df.sort_values('mape_mean')
    
    # Save summary
    summary_file = os.path.join(output_dir, 'summary_by_model.csv')
    summary_df.to_csv(summary_file, index=False)
    print(f"\n\nSummary saved to: {summary_file}")
    
    # Save detailed results
    detailed_file = os.path.join(output_dir, 'detailed_results.csv')
    df.to_csv(detailed_file, index=False)
    print(f"Detailed results saved to: {detailed_file}")
    
    # Generate model comparison table
    print("\n" + "="*80)
    print("MODEL COMPARISON (sorted by MAPE - lower is better)")
    print("="*80)
    
    if 'mape_mean' in summary_df.columns:
        comparison_df = summary_df[['model', 'count', 'mape_mean', 'mape_std', 
                                    'rmse_mean', 'r2_score_mean']].copy()
        comparison_df.columns = ['Model', 'Workloads', 'MAPE (mean)', 'MAPE (std)',
                                'RMSE (mean)', 'R² (mean)']
        print(comparison_df.to_string(index=False))
        
        comparison_file = os.path.join(output_dir, 'model_comparison.csv')
        comparison_df.to_csv(comparison_file, index=False)
        print(f"\nComparison table saved to: {comparison_file}")
    
    # Generate per-workload results for best model
    if 'mape' in df.columns:
        print("\n" + "="*80)
        print("TOP 10 BEST PREDICTIONS (lowest MAPE)")
        print("="*80)
        
        best_predictions = df.nsmallest(10, 'mape')[['workload', 'model', 'mape', 'rmse', 'r2_score']]
        print(best_predictions.to_string(index=False))
        
        print("\n" + "="*80)
        print("TOP 10 WORST PREDICTIONS (highest MAPE)")
        print("="*80)
        
        worst_predictions = df.nlargest(10, 'mape')[['workload', 'model', 'mape', 'rmse', 'r2_score']]
        print(worst_predictions.to_string(index=False))
    
    # Create a text summary report
    report_file = os.path.join(output_dir, 'summary_report.txt')
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("WORKLOAD FORECASTING RESULTS SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Input Directory: {output_dir}\n")
        f.write(f"Total Experiments: {len(df)}\n")
        f.write(f"Models Tested: {', '.join(sorted(df['model'].unique()))}\n")
        f.write(f"Unique Workloads: {df['workload'].nunique()}\n\n")
        
        f.write("="*80 + "\n")
        f.write("SUMMARY BY MODEL\n")
        f.write("="*80 + "\n\n")
        
        for model in sorted(df['model'].unique()):
            model_df = df[df['model'] == model]
            f.write(f"{model.upper()}\n")
            f.write("-" * 80 + "\n")
            f.write(f"Number of workloads: {len(model_df)}\n")
            
            for metric in available_metrics:
                if metric in model_df.columns:
                    values = model_df[metric].dropna()
                    if len(values) > 0:
                        mean_val = values.mean()
                        std_val = values.std()
                        median_val = values.median()
                        f.write(f"  {metric.upper():15s}: {mean_val:8.4f} ± {std_val:7.4f}  "
                              f"(median: {median_val:8.4f})\n")
            f.write("\n")
        
        if 'mape_mean' in summary_df.columns:
            f.write("="*80 + "\n")
            f.write("MODEL RANKING (by MAPE - lower is better)\n")
            f.write("="*80 + "\n\n")
            
            for idx, row in summary_df.iterrows():
                rank = idx + 1
                best_marker = " ← BEST" if rank == 1 else (" ← WORST" if rank == len(summary_df) else "")
                f.write(f"{rank}. {row['model']:15s}  MAPE: {row['mape_mean']:8.4f} ± {row['mape_std']:7.4f}{best_marker}\n")
    
    print(f"\nText summary saved to: {report_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Process workload forecasting log files and extract metrics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process BUS_CYCLES logs
  %(prog)s --input counter_forecasting_logs/BUS_CYCLES \\
           --output counter_forecasting_logs/BUS_CYCLES

  # Process with custom output directory
  %(prog)s --input logs/ --output results/summary/
        """
    )
    
    parser.add_argument('--input', '-i', required=True,
                       help='Input directory containing log files')
    parser.add_argument('--output', '-o', required=True,
                       help='Output directory for summary files')
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.isdir(args.input):
        print(f"Error: Input directory '{args.input}' does not exist")
        return 1
    
    # Process logs
    df = process_directory(args.input)
    
    if df is None or len(df) == 0:
        print("No valid results found")
        return 1
    
    # Generate summary
    generate_summary(df, args.output)
    
    print("\n" + "="*80)
    print("Processing complete!")
    print("="*80)
    
    return 0


if __name__ == '__main__':
    exit(main())


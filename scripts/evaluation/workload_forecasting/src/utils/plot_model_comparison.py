#!/usr/bin/env python3
"""
Plot model comparison bar graphs for workload forecasting results.
Creates 4 subplots: SPEC@1.5GHz, SPEC@3.0GHz, DaCapo@1.5GHz, DaCapo@3.0GHz
Each subplot shows 5 bars per workload (dt, mlp, lstm, stacked_lstm, transformer)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

def load_and_process_data(csv_path):
    """Load the MAPE comparison CSV and process it."""
    df = pd.read_csv(csv_path)
    
    # Check if stacked_lstm column already exists
    if 'stacked_lstm' in df.columns:
        # Data is already in the correct format
        # Just ensure we have the correct columns
        required_cols = ['workload', 'dt', 'mlp', 'lstm', 'transformer', 'stacked_lstm']
        
        # Filter out any rows with '_stacked' suffix (if they exist)
        df_clean = df[~df['workload'].str.contains('_stacked', na=False)].copy()
        
        # Return with just the required columns
        available_cols = [col for col in required_cols if col in df_clean.columns]
        return df_clean[available_cols]
    
    # Process stacked_lstm results (old format)
    # The stacked_lstm results are in rows with "_stacked" suffix
    results = []
    
    for idx, row in df.iterrows():
        workload = row['workload']
        
        # Check if this is a stacked_lstm row
        if '_stacked' in workload:
            # Skip, we'll handle these separately
            continue
        
        # Get base model results
        result = {
            'workload': workload,
            'dt': row.get('dt', np.nan),
            'mlp': row.get('mlp', np.nan),
            'lstm': row.get('lstm', np.nan),
            'transformer': row.get('transformer', np.nan),
            'stacked_lstm': np.nan
        }
        
        # Look for corresponding stacked_lstm result
        stacked_workload = workload + '_stacked'
        stacked_rows = df[df['workload'] == stacked_workload]
        if not stacked_rows.empty:
            result['stacked_lstm'] = stacked_rows.iloc[0].get('lstm', np.nan)
        
        results.append(result)
    
    return pd.DataFrame(results)

def separate_by_benchmark_and_frequency(df):
    """Separate data into SPEC/DaCapo and 1.5GHz/3.0GHz."""
    data = {
        'spec_1.5': [],
        'spec_3.0': [],
        'dacapo_1.5': [],
        'dacapo_3.0': []
    }
    
    for idx, row in df.iterrows():
        workload = row['workload']
        
        # Determine benchmark type and frequency
        is_spec = workload.startswith('spec_')
        is_dacapo = workload.startswith('dacapo_')
        is_15ghz = '1.5GHz' in workload
        is_30ghz = '3.0GHz' in workload
        
        # Clean workload name (remove frequency suffix and benchmark prefix for display)
        clean_name = workload.replace('_1.5GHz_merged', '').replace('_3.0GHz_merged', '')
        clean_name = clean_name.replace('spec_', '').replace('dacapo_', '')
        
        row_data = {
            'workload': clean_name,
            'dt': row['dt'],
            'mlp': row['mlp'],
            'lstm': row['lstm'],
            'stacked_lstm': row['stacked_lstm'],
            'transformer': row['transformer']
        }
        
        if is_spec and is_15ghz:
            data['spec_1.5'].append(row_data)
        elif is_spec and is_30ghz:
            data['spec_3.0'].append(row_data)
        elif is_dacapo and is_15ghz:
            data['dacapo_1.5'].append(row_data)
        elif is_dacapo and is_30ghz:
            data['dacapo_3.0'].append(row_data)
    
    # Convert to DataFrames
    for key in data:
        data[key] = pd.DataFrame(data[key])
    
    return data

def plot_comparison_bars(ax, df, title, ylabel='MAPE (%)', max_y=None):
    """Plot grouped bar chart for model comparison."""
    if df.empty:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return
    
    models = ['dt', 'mlp', 'lstm', 'stacked_lstm', 'transformer']
    model_colors = {
        'dt': '#1f77b4',           # Blue
        'mlp': '#ff7f0e',          # Orange
        'lstm': '#2ca02c',         # Green
        'stacked_lstm': '#d62728', # Red
        'transformer': '#9467bd'   # Purple
    }
    
    workloads = df['workload'].values
    n_workloads = len(workloads)
    n_models = len(models)
    
    # Calculate average for each model
    model_averages = {}
    for model in models:
        model_averages[model] = df[model].mean()
    
    # Set up bar positions (including space for average)
    bar_width = 0.15
    x = np.arange(n_workloads + 1)  # +1 for average bar
    
    # Plot bars for each model
    for i, model in enumerate(models):
        offset = (i - n_models/2 + 0.5) * bar_width
        values = df[model].values
        
        # Add average to values
        values_with_avg = np.append(values, model_averages[model])
        
        # Replace NaN with 0 for plotting (but keep track)
        plot_values = np.nan_to_num(values_with_avg, nan=0)
        
        bars = ax.bar(x + offset, plot_values, bar_width, 
                     label=model.replace('_', ' ').title(),
                     color=model_colors[model],
                     alpha=0.8)
        
        # Mark bars with missing data (only for non-average bars)
        for j, (val, plot_val) in enumerate(zip(values, plot_values[:-1])):
            if np.isnan(val):
                ax.text(x[j] + offset, plot_val, 'N/A', 
                       ha='center', va='bottom', fontsize=6, rotation=90)
    
    # Customize plot
    ax.set_xlabel('Workload', fontsize=11, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    
    # Set x-ticks with workload names and "Average"
    workload_labels = list(workloads) + ['Average']
    ax.set_xticks(x)
    ax.set_xticklabels(workload_labels, rotation=45, ha='right', fontsize=9)
    
    ax.legend(loc='upper right', fontsize=9, ncol=1, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Set y-axis limit if provided
    if max_y is not None:
        ax.set_ylim(0, max_y)
    
    # Add horizontal line at y=5 for reference (reasonable MAPE target)
    ax.axhline(y=5, color='gray', linestyle=':', linewidth=1, alpha=0.5)

def create_individual_plot(df, title, output_path, max_y=None):
    """Create a single plot and save as SVG."""
    fig, ax = plt.subplots(1, 1, figsize=(18, 4))
    
    plot_comparison_bars(ax, df, title, max_y=max_y)
    
    plt.tight_layout()
    
    print(f"Saving plot to {output_path}...")
    plt.savefig(output_path, format='svg', bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved: {output_path}")

def create_comparison_plot(csv_path, output_dir=None):
    """Create 4 separate SVG plots for model comparison."""
    print(f"Loading data from {csv_path}...")
    df = load_and_process_data(csv_path)
    
    print("Separating data by benchmark and frequency...")
    data = separate_by_benchmark_and_frequency(df)
    
    # Find global max for consistent y-axis
    all_values = []
    for key in data:
        for model in ['dt', 'mlp', 'lstm', 'stacked_lstm', 'transformer']:
            all_values.extend(data[key][model].dropna().values)
    global_max = np.max(all_values) * 1.1 if all_values else 30
    
    # Determine output directory
    if output_dir is None:
        output_dir = os.path.dirname(csv_path)
    
    # Create individual plots
    print("\nCreating individual SVG plots...")
    
    create_individual_plot(data['spec_1.5'], 
                          'SPEC Benchmarks @ 1.5GHz',
                          os.path.join(output_dir, 'spec_1.5GHz_comparison.svg'),
                          max_y=global_max)
    
    create_individual_plot(data['spec_3.0'], 
                          'SPEC Benchmarks @ 3.0GHz',
                          os.path.join(output_dir, 'spec_3.0GHz_comparison.svg'),
                          max_y=global_max)
    
    create_individual_plot(data['dacapo_1.5'], 
                          'DaCapo Benchmarks @ 1.5GHz',
                          os.path.join(output_dir, 'dacapo_1.5GHz_comparison.svg'),
                          max_y=global_max)
    
    create_individual_plot(data['dacapo_3.0'], 
                          'DaCapo Benchmarks @ 3.0GHz',
                          os.path.join(output_dir, 'dacapo_3.0GHz_comparison.svg'),
                          max_y=global_max)
    
    print(f"\nAll plots saved to: {output_dir}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("Summary Statistics (MAPE %)")
    print("="*80)
    for key, df_subset in data.items():
        if not df_subset.empty:
            print(f"\n{key.upper().replace('_', ' @ ').replace('.', '.')}GHz:")
            for model in ['dt', 'mlp', 'lstm', 'stacked_lstm', 'transformer']:
                values = df_subset[model].dropna()
                if len(values) > 0:
                    print(f"  {model:15s}: Mean={values.mean():6.2f}, "
                          f"Median={values.median():6.2f}, "
                          f"Std={values.std():6.2f}, "
                          f"Min={values.min():6.2f}, "
                          f"Max={values.max():6.2f}")

def main():
    # Default paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_csv = os.path.join(script_dir, 'parallel_arm_results', 'mape_comparison_latest.csv')
    default_output_dir = os.path.join(script_dir, 'parallel_arm_results')
    
    # Parse arguments
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = default_csv
    
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    else:
        output_dir = default_output_dir
    
    # Check if CSV exists
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found: {csv_path}")
        print(f"\nUsage: {sys.argv[0]} [csv_path] [output_directory]")
        print(f"  Generates 4 separate SVG files:")
        print(f"    - spec_1.5GHz_comparison.svg")
        print(f"    - spec_3.0GHz_comparison.svg")
        print(f"    - dacapo_1.5GHz_comparison.svg")
        print(f"    - dacapo_3.0GHz_comparison.svg")
        sys.exit(1)
    
    # Create plots
    create_comparison_plot(csv_path, output_dir)

if __name__ == "__main__":
    main()


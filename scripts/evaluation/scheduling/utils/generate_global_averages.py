import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
from pathlib import Path
import argparse
import re

def plot_subset(plot_df, title, out_file, metric, baseline_policy='Proactive_Hetero_Oracle', y_limits=None, show_values=False):
    """Helper function to plot a specific subset of policies, dynamically re-normalizing to a chosen baseline."""
    if plot_df.empty or len(plot_df) <= 1: return
    plot_df = plot_df.copy()
    
    baseline_row = plot_df[plot_df['Policy'] == baseline_policy]
    if not baseline_row.empty:
        base_val = baseline_row['Normalized_Score'].values[0]
        plot_df['Normalized_Score'] = plot_df['Normalized_Score'] / base_val
    else:
        base_val = plot_df['Normalized_Score'].min()
        plot_df['Normalized_Score'] = plot_df['Normalized_Score'] / base_val
        
    plot_df = plot_df.sort_values(by='Normalized_Score')
    
    if not baseline_row.empty:
        other_rows = plot_df[plot_df['Policy'] != baseline_policy]
        baseline_row_updated = plot_df[plot_df['Policy'] == baseline_policy]
        plot_df = pd.concat([baseline_row_updated, other_rows])
    
    policies = plot_df['Policy'].tolist()
    scores = plot_df['Normalized_Score'].tolist()
    
    cmap = plt.get_cmap('tab20')
    colors = [cmap(i % 20) for i in range(len(policies))]
    
    plt.figure(figsize=(12, 7))
    display_keys = [p.replace('_', '\n') for p in policies]
    
    bars = plt.bar(display_keys, scores, color=colors, edgecolor='black')
    
    if show_values:
        for bar in bars:
            yval = bar.get_height()
            if y_limits and yval > y_limits[1]:
                text_y = y_limits[1] - ((y_limits[1] - y_limits[0]) * 0.05)
                plt.text(bar.get_x() + bar.get_width() / 2.0, text_y, f"{yval:.2f}", 
                         ha='center', va='top', fontsize=10, fontweight='bold', color='red',
                         bbox=dict(facecolor='white', alpha=0.9, edgecolor='red', boxstyle='round,pad=0.2'))
            else:
                text_y = yval + 0.005
                plt.text(bar.get_x() + bar.get_width() / 2.0, text_y, f"{yval:.2f}", 
                         ha='center', va='bottom', fontsize=10, fontweight='bold')

    display_baseline_name = baseline_policy.replace('_', ' ')
    plt.axhline(1.0, color='red', linestyle='--', linewidth=2, label=f'1.0 = {display_baseline_name}')
    
    if y_limits: plt.ylim(y_limits)
        
    plt.title(title)
    plt.ylabel(f"Relative {metric} Score\n(Normalized to {display_baseline_name})")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(out_file, dpi=100)
    plt.close()


def plot_phase_grouped_bar(df, title, out_file, policies, baseline_policy, y_limits=None):
    """Generates a grouped bar chart with every Phase on the X-axis and specific Policies as sub-bars."""
    
    plot_df = df[df['Policy'].isin(policies)].copy()
    if plot_df.empty: return
    
    # Pivot so Workload_Phase strings are rows and Policies are columns
    pivot_df = plot_df.pivot_table(index='Workload_Phase', columns='Policy', values='Normalized_Score', aggfunc='mean')
    
    if baseline_policy in pivot_df.columns:
        pivot_df = pivot_df.dropna(subset=[baseline_policy])
    else:
        return

    valid_cols = [p for p in policies if p in pivot_df.columns]
    pivot_df = pivot_df[valid_cols]
    
    # STRICT LOCAL NORMALIZATION per individual phase!
    for col in pivot_df.columns:
        pivot_df[col] = pivot_df[col] / pivot_df[baseline_policy]
        
    # Add the Overall Average Row
    avg_row = pivot_df.mean()
    pivot_df.loc['AVERAGE'] = avg_row
            
    # We use a very wide figure (24x7) because we might have dozens of phases + the average
    ax = pivot_df.plot(kind='bar', figsize=(24, 7), width=0.8, colormap='viridis', edgecolor='black')
    
    plt.axhline(1.0, color='red', linestyle='--', linewidth=2, label=f'1.0 = {baseline_policy}')
    
    if y_limits: plt.ylim(y_limits)
        
    # Outlier handling for massive penalties
    if y_limits:
        for p in ax.patches:
            val = p.get_height()
            if val > y_limits[1]:
                text_y = y_limits[1] - ((y_limits[1] - y_limits[0]) * 0.05)
                ax.text(p.get_x() + p.get_width() / 2.0, text_y, 
                        f'{val:.1f}', ha='center', va='top', rotation=90, color='red',
                        fontsize=10, fontweight='bold', 
                        bbox=dict(facecolor='white', alpha=0.9, edgecolor='red', pad=1))

    plt.title(title, fontsize=16, fontweight='bold')
    plt.ylabel(f"Relative EDP Score\n(Normalized strictly to {baseline_policy})", fontsize=12)
    plt.xlabel("DaCapo Workload Phase (Final column is Overall Average)", fontsize=12)
    
    # Highlight the AVERAGE label
    xticks = ax.get_xticklabels()
    for tick in xticks:
        if tick.get_text() == 'AVERAGE':
            tick.set_fontweight('bold')
            tick.set_color('red')
            
    plt.xticks(rotation=45, ha='right')
    plt.legend(title="Scheduling/DVFS Policy", bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Aggregate phase CSVs into global policy averages and specific subsets.")
    parser.add_argument('--csv_dir', type=str, required=True, help="Path to the csv_data/ directory")
    parser.add_argument('--out_dir', type=str, required=True, help="Directory to save the summaries")
    args = parser.parse_args()

    csv_path = Path(args.csv_dir)
    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    all_data = []
    search_pattern = csv_path / "bar_*.csv"
    files = glob.glob(str(search_pattern))
    
    if not files:
        print(f"No bar_*.csv files found in {csv_path}")
        return
        
    print(f"Found {len(files)} phase files. Aggregating...")

    file_pattern = re.compile(r"bar_(.+?)_(.+?)_(.+)_phase(\d+)\.csv")

    for f in files:
        basename = Path(f).name
        match = file_pattern.match(basename)
        if match:
            category, m_type, wl, ph = match.groups()
        else:
            continue
        
        df = pd.read_csv(f)
        df = df.rename(columns={f'Final_{m_type}': 'Raw_Score', f'Normalized_{m_type}': 'Normalized_Score'})
        df['Category'] = category
        df['Metric'] = m_type
        df['Workload'] = wl
        df['Phase'] = ph
        all_data.append(df)

    master_df = pd.concat(all_data, ignore_index=True)

    # ==========================================
    # 1. SAVE GLOBAL AND PHASE CSV SUMMARIES
    # ==========================================
    # Global average across everything
    global_summary = master_df.groupby(['Category', 'Metric', 'Policy'])['Normalized_Score'].mean().reset_index()
    global_summary = global_summary.sort_values(by=['Category', 'Metric', 'Normalized_Score'])
    global_summary.to_csv(out_path / "global_policy_averages.csv", index=False)
    
    # Phase-specific averages (NEW)
    phase_summary = master_df.groupby(['Category', 'Metric', 'Workload', 'Phase', 'Policy'])['Normalized_Score'].mean().reset_index()
    
    # Create the combined X-Axis label string: e.g., "tomcat (Ph 1)"
    phase_summary['Workload_Phase'] = phase_summary['Workload'] + " (Ph " + phase_summary['Phase'].astype(str) + ")"
    phase_summary.to_csv(out_path / "phase_policy_averages.csv", index=False)

    # ==========================================
    # 2. GENERATE THE PER-PHASE CLUSTER CHARTS
    # ==========================================
    edp_ph_df = phase_summary[phase_summary['Metric'] == 'EDP']
    
    e_core_policies = ['Static_E_2.0GHz', 'Global_Oracle_E', 'Reactive_1_Step_E', 'Proactive_1_Step_E', 'Proactive_32_Step_E']
    plot_phase_grouped_bar(edp_ph_df, 
                           "Per-Phase EDP Comparison (E-Core strictly vs Global_Oracle_E)", 
                           out_path / "phase_cluster_ecore_edp.png", 
                           e_core_policies, 
                           baseline_policy='Global_Oracle_E',
                           y_limits=(0.8, 3.0)) 
                              
    p_core_policies = ['Static_P_2.0GHz', 'Global_Oracle_P', 'Reactive_1_Step_P', 'Proactive_1_Step_P', 'Proactive_32_Step_P']
    plot_phase_grouped_bar(edp_ph_df, 
                           "Per-Phase EDP Comparison (P-Core strictly vs Global_Oracle_P)", 
                           out_path / "phase_cluster_pcore_edp.png", 
                           p_core_policies, 
                           baseline_policy='Global_Oracle_P',
                           y_limits=(0.8, 1.5)) 

    # ==========================================
    # 3. GENERATE SPECIFIC FAIR-BASELINE PLOTS
    # ==========================================
    metrics = global_summary['Metric'].unique()
    for metric in metrics:
        metric_df = global_summary[global_summary['Metric'] == metric]
        
        e_core_df = metric_df[metric_df['Category'] == 'EDVFS']
        plot_subset(e_core_df, f"DVFS Only (E-Core) - {metric}", out_path / f"custom_dvfs_ecore_{metric}.png", metric, 'Proactive_E_Oracle', (0.8, 1.25), True)     
        
        p_core_df = metric_df[metric_df['Category'] == 'PDVFS']
        plot_subset(p_core_df, f"DVFS Only (P-Core) - {metric}", out_path / f"custom_dvfs_pcore_{metric}.png", metric, 'Proactive_P_Oracle', (0.8, 1.25), True)
                    
        for freq in ['1.0GHz', '2.0GHz', '3.0GHz', '4.0GHz']:
            freq_df = metric_df[metric_df['Policy'].str.contains(freq)]
            plot_subset(freq_df, f"Iso-Frequency Heterogeneous ({freq}) - {metric}", out_path / f"custom_hetero_iso_{freq.replace('.', '_')}_{metric}.png", metric, f'Global_Oracle_Hetero_{freq}', (0.8, 1.25), True)

    for (cat, metric), group_df in global_summary.groupby(['Category', 'Metric']):
        overall_baseline = 'Global_Oracle_Combined' if cat == 'COMBINED' else 'Proactive_Hetero_Oracle'
        plot_subset(group_df, f"GLOBAL AVERAGE: {cat} - {metric}", out_path / f"global_avg_{cat}_{metric}.png", metric, overall_baseline)

    print(f"Generated global, custom subset, and ALL PHASE plots in {out_path}")

if __name__ == "__main__":
    main()
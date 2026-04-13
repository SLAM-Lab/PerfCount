import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

def plot_phase_grouped_bar(df, title, out_file, policies, baseline_policy, y_limits=None):
    """Generates a grouped bar chart with Phases on the X-axis and specific Policies as sub-bars."""
    
    # 1. Filter and Pivot
    plot_df = df[df['Policy'].isin(policies)].copy()
    if plot_df.empty:
        print(f"No data found for policies: {policies}")
        return
        
    # FIX: Pivot on 'Workload_Phase' instead of 'Workload'
    pivot_df = plot_df.pivot_table(index='Workload_Phase', columns='Policy', values='Normalized_Score', aggfunc='mean')
    
    # Drop any workloads that somehow didn't run the baseline policy
    if baseline_policy in pivot_df.columns:
        pivot_df = pivot_df.dropna(subset=[baseline_policy])
    else:
        print(f"Error: Baseline policy {baseline_policy} not found in data.")
        return

    # Reorder columns to strictly match the requested list
    valid_cols = [p for p in policies if p in pivot_df.columns]
    pivot_df = pivot_df[valid_cols]
    
    # 2. STRICT LOCAL NORMALIZATION
    # Divide every column by the local Oracle for that specific phase
    for col in pivot_df.columns:
        pivot_df[col] = pivot_df[col] / pivot_df[baseline_policy]
        
    # --- NEW: ADD AVERAGE ROW ---
    # Calculate the mean across all phases for each policy and append it as the final row
    avg_row = pivot_df.mean()
    pivot_df.loc['AVERAGE'] = avg_row
            
    # 3. Plotting
    # Bumped width slightly to 24 to give plenty of room for all the phases!
    ax = pivot_df.plot(kind='bar', figsize=(24, 7), width=0.8, colormap='viridis', edgecolor='black')
    
    # Add the red 1.0 Baseline reference line
    plt.axhline(1.0, color='red', linestyle='--', linewidth=2, label=f'1.0 = {baseline_policy}')
    
    if y_limits: 
        plt.ylim(y_limits)
        
    # 4. Outlier Text Labels
    # If a bar shoots past the ceiling, write its true value inside the top edge of the graph
    if y_limits:
        for p in ax.patches:
            val = p.get_height()
            if val > y_limits[1]:
                # Place text slightly below the ceiling
                text_y = y_limits[1] - ((y_limits[1] - y_limits[0]) * 0.05)
                ax.text(p.get_x() + p.get_width() / 2.0, text_y, 
                        f'{val:.1f}', ha='center', va='top', rotation=90, color='red',
                        fontsize=10, fontweight='bold', 
                        bbox=dict(facecolor='white', alpha=0.9, edgecolor='red', pad=1))

    plt.title(title, fontsize=16, fontweight='bold')
    plt.ylabel(f"Relative EDP Score\n(Normalized strictly to {baseline_policy})", fontsize=12)
    plt.xlabel("DaCapo Workload Phase (Final column is Overall Average)", fontsize=12)
    
    # Bold the 'AVERAGE' label on the X-axis so it stands out
    xticks = ax.get_xticklabels()
    for tick in xticks:
        if tick.get_text() == 'AVERAGE':
            tick.set_fontweight('bold')
            tick.set_color('red')
            
    plt.xticks(rotation=45, ha='right')
    
    # Put legend safely outside the plot
    plt.legend(title="Scheduling/DVFS Policy", bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_file}")

def plot_global_average_bar(df, title, out_file, policies, baseline_policy, y_limits=None):
    """Generates a single bar chart showing the global average EDP for all provided policies."""
    plot_df = df[df['Policy'].isin(policies)].copy()
    if plot_df.empty:
        print(f"No data found for global average policies: {policies}")
        return
        
    pivot_df = plot_df.pivot_table(index='Workload_Phase', columns='Policy', values='Normalized_Score', aggfunc='mean')
    
    if baseline_policy in pivot_df.columns:
        pivot_df = pivot_df.dropna(subset=[baseline_policy])
    else:
        print(f"Error: Baseline policy {baseline_policy} not found in data.")
        return

    valid_cols = [p for p in policies if p in pivot_df.columns]
    pivot_df = pivot_df[valid_cols]
    
    for col in pivot_df.columns:
        pivot_df[col] = pivot_df[col] / pivot_df[baseline_policy]
        
    avg_series = pivot_df.mean()
    
    plt.figure(figsize=(14, 7))
    ax = avg_series.plot(kind='bar', color='steelblue', edgecolor='black', zorder=3)
    
    plt.axhline(1.0, color='red', linestyle='--', linewidth=2, label=f'1.0 = {baseline_policy}', zorder=4)
    
    if y_limits:
        plt.ylim(y_limits)
        
    # Annotate bars
    for i, p in enumerate(ax.patches):
        val = p.get_height()
        # Add values on top of bars
        ax.text(p.get_x() + p.get_width() / 2.0, val + 0.02, f'{val:.2f}', 
                ha='center', va='bottom', fontweight='bold', fontsize=11)
                
    plt.title(title, fontsize=16, fontweight='bold')
    plt.ylabel(f"Average Relative EDP Score\n(Normalized to {baseline_policy})", fontsize=12)
    plt.xlabel("Policy", fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=11)
    
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
    plt.tight_layout()
    
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate per-phase grouped bar charts from the summary CSV.")
    parser.add_argument('--csv_file', type=str, required=True, help="Path to phase_policy_averages.csv")
    parser.add_argument('--out_dir', type=str, required=True, help="Directory to save the generated plots")
    args = parser.parse_args()

    csv_path = Path(args.csv_file)
    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        print(f"Could not find {csv_path}")
        return

    # Load the pre-aggregated data
    df = pd.read_csv(csv_path)
    
    # Filter down to just the EDP metric
    edp_df = df[df['Metric'] == 'EDP']

    # --- Chart 1: E-Core Drilldown ---
    e_core_policies = ['Static_E_2.0GHz', 'Proactive_E_Oracle', 'Reactive_1_Step_E', 'Proactive_1_Step_E']
    plot_phase_grouped_bar(edp_df, 
                              "Per-Phase EDP Comparison (E-Core strictly vs Proactive_E_Oracle)", 
                              out_path / "phase_cluster_ecore_edp.png", 
                              e_core_policies, 
                              baseline_policy='Proactive_E_Oracle',
                              y_limits=(0.8, 1.65)) 
                              
    # --- Chart 2: P-Core Drilldown ---
    p_core_policies = ['Static_P_2.0GHz', 'Proactive_P_Oracle', 'Reactive_1_Step_P', 'Proactive_1_Step_P', 'Proactive_N_Step_P_Window_5', 'Proactive_N_Step_P_Window_10', 'Linux_Schedutil_Proxy', 'Intel_HWP_Proxy']
    plot_phase_grouped_bar(edp_df, 
                              "Per-Phase EDP Comparison (P-Core strictly vs Proactive_P_Oracle)", 
                              out_path / "phase_cluster_pcore_edp.png", 
                              p_core_policies, 
                              baseline_policy='Proactive_P_Oracle',
                              y_limits=(0.8, 1.65))

    # --- Chart 3: ALL E-Core Global Average ---
    e_core_all_policies = [
        'Static_E_1.0GHz', 'Static_E_2.0GHz', 'Static_E_3.0GHz', 'Static_E_4.0GHz', 
        'Reactive_1_Step_E', 'Proactive_1_Step_E', 
        'Proactive_N_Step_E_Window_5', 'Proactive_N_Step_E_Window_10', 
        'Proactive_E_Oracle'
    ]
    plot_global_average_bar(edp_df,
                            "Global Average EDP Comparison (All E-Core Policies)",
                            out_path / "global_avg_ecore_edp.png",
                            e_core_all_policies,
                            baseline_policy='Proactive_E_Oracle',
                            y_limits=None)

    # --- Chart 4: ALL P-Core Global Average ---
    p_core_all_policies = [
        'Static_P_1.0GHz', 'Static_P_2.0GHz', 'Static_P_3.0GHz', 'Static_P_4.0GHz', 
        'Reactive_1_Step_P', 'Proactive_1_Step_P', 
        'Proactive_N_Step_P_Window_5', 'Proactive_N_Step_P_Window_10', 
        'Linux_Schedutil_Proxy', 'Intel_HWP_Proxy',
        'Proactive_P_Oracle'
    ]
    plot_global_average_bar(edp_df,
                            "Global Average EDP Comparison (All P-Core Policies)",
                            out_path / "global_avg_pcore_edp.png",
                            p_core_all_policies,
                            baseline_policy='Proactive_P_Oracle',
                            y_limits=None)

if __name__ == "__main__":
    main()
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

def plot_phase_grouped_bar(df, title, out_file, policies, baseline_policy, y_limits=None):
    """Generates a grouped bar chart with Phases on the X-axis and specific Policies as sub-bars."""
    
    # 1. Filter and Pivot
    plot_df = df[df['Policy'].isin(policies)].copy()
    if plot_df.empty:
        print(f"No data found for phase chart {out_file.name}")
        return
        
    # Pivot on 'Workload_Phase'
    pivot_df = plot_df.pivot_table(index='Workload_Phase', columns='Policy', values='Normalized_Score', aggfunc='mean')
    
    # Drop any workloads that somehow didn't run the baseline policy
    if baseline_policy in pivot_df.columns:
        pivot_df = pivot_df.dropna(subset=[baseline_policy])
    else:
        print(f"Error: Baseline policy {baseline_policy} not found in data for {out_file.name}.")
        return

    # Reorder columns to strictly match the requested list
    valid_cols = [p for p in policies if p in pivot_df.columns]
    pivot_df = pivot_df[valid_cols]
    
    # 2. STRICT LOCAL NORMALIZATION
    for col in pivot_df.columns:
        pivot_df[col] = pivot_df[col] / pivot_df[baseline_policy]
        
    # --- ADD AVERAGE ROW ---
    avg_row = pivot_df.mean()
    pivot_df.loc['AVERAGE'] = avg_row
            
    # 3. Plotting
    ax = pivot_df.plot(kind='bar', figsize=(24, 7), width=0.8, colormap='viridis', edgecolor='black')
    
    #plt.axhline(1.0, color='red', linestyle='--', linewidth=2, label=f'1.0 = {baseline_policy}')
    
    if y_limits: 
        plt.ylim(y_limits)
        
    # 4. Outlier Text Labels
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
    plt.xlabel("DaCapo / SPEC Workload Phase (Final column is Overall Average)", fontsize=12)
    
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
    print(f"Saved: {out_file.name}")


def plot_global_average_bar(df, title, out_file, policies, baseline_policy, y_limits=None):
    """Generates a single bar chart showing the global average EDP for all provided policies."""
    plot_df = df[df['Policy'].isin(policies)].copy()
    if plot_df.empty:
        print(f"No data found for global average chart {out_file.name}")
        return
        
    pivot_df = plot_df.pivot_table(index='Workload_Phase', columns='Policy', values='Normalized_Score', aggfunc='mean')
    
    if baseline_policy in pivot_df.columns:
        pivot_df = pivot_df.dropna(subset=[baseline_policy])
    else:
        print(f"Error: Baseline policy {baseline_policy} not found in data for {out_file.name}.")
        return

    valid_cols = [p for p in policies if p in pivot_df.columns]
    pivot_df = pivot_df[valid_cols]
    
    for col in pivot_df.columns:
        pivot_df[col] = pivot_df[col] / pivot_df[baseline_policy]
        
    avg_series = pivot_df.mean()
    
    plt.figure(figsize=(14, 7))
    ax = avg_series.plot(kind='bar', color='steelblue', edgecolor='black', zorder=3)
    
    #plt.axhline(1.0, color='red', linestyle='--', linewidth=2, label=f'1.0 = {baseline_policy}', zorder=4)
    
    if y_limits:
        plt.ylim(y_limits)
        
    for i, p in enumerate(ax.patches):
        val = p.get_height()
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
    print(f"Saved: {out_file.name}")


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

    df = pd.read_csv(csv_path)
    edp_df = df[df['Metric'] == 'ED2P']

    # --- SEPARATE WORKLOADS INTO DACAPO AND SPEC ---
    dacapo_df = edp_df[edp_df['Workload'].str.contains('dacapo', case=False, na=False)]
    spec_df = edp_df[~edp_df['Workload'].str.contains('dacapo', case=False, na=False)]

    # ADDED: Proactive_N_Step_E_Window_5 to the E-core phase policies
    e_core_phase_policies = [
        #'Static_E_2.0GHz', 
        'Proactive_E_Oracle', 
        'Reactive_1_Step_E', 
        'Proactive_1_Step_E', 
        'Proactive_N_Step_E_Window_5'
    ]
    
    # ENSURED: Window 5 and Window 10 are grouped logically
    p_core_phase_policies = [
        #'Static_P_2.0GHz', 
        'Proactive_P_Oracle', 
        'Reactive_1_Step_P', 
        'Proactive_1_Step_P', 
        'Proactive_N_Step_P_Window_5', 
        #'Proactive_N_Step_P_Window_10', 
        #'Linux_Schedutil_Proxy', 'Intel_HWP_Proxy'
    ]
    
    e_core_all_policies = [
        'Static_E_1.0GHz', 'Static_E_2.0GHz', 'Static_E_3.0GHz', 'Static_E_4.0GHz', 
        'Reactive_1_Step_E', 'Proactive_1_Step_E', 
        'Proactive_N_Step_E_Window_5', 'Proactive_N_Step_E_Window_10', 
        'Proactive_E_Oracle'
    ]
    
    p_core_all_policies = [
        'Static_P_1.0GHz', 'Static_P_2.0GHz', 'Static_P_3.0GHz', 'Static_P_4.0GHz', 
        'Reactive_1_Step_P', 'Proactive_1_Step_P', 
        'Proactive_N_Step_P_Window_5', 'Proactive_N_Step_P_Window_10', 
        'Linux_Schedutil_Proxy', 'Intel_HWP_Proxy',
        'Proactive_P_Oracle'
    ]

    # ===============================================
    # 1. PER-PHASE DRILLDOWN PLOTS (4 Figures)
    # ===============================================
    plot_phase_grouped_bar(dacapo_df, "Per-Phase EDDP Comparison (E-Core strictly vs Oracle) - DACAPO", 
                           out_path / "phase_cluster_ecore_dacapo_eddp.png", e_core_phase_policies, 
                           baseline_policy='Proactive_E_Oracle', y_limits=(0.8, 1.25)) 
                           
    plot_phase_grouped_bar(spec_df, "Per-Phase EDDP Comparison (E-Core strictly vs Oracle) - SPEC", 
                           out_path / "phase_cluster_ecore_spec_eddp.png", e_core_phase_policies, 
                           baseline_policy='Proactive_E_Oracle', y_limits=(0.8, 1.25)) 

    plot_phase_grouped_bar(dacapo_df, "Per-Phase EDDP Comparison (P-Core strictly vs Oracle) - DACAPO", 
                           out_path / "phase_cluster_pcore_dacapo_eddp.png", p_core_phase_policies, 
                           baseline_policy='Proactive_P_Oracle', y_limits=(0.8, 1.25))

    plot_phase_grouped_bar(spec_df, "Per-Phase EDDP Comparison (P-Core strictly vs Oracle) - SPEC", 
                           out_path / "phase_cluster_pcore_spec_eddp.png", p_core_phase_policies, 
                           baseline_policy='Proactive_P_Oracle', y_limits=(0.8, 1.25))

    # ===============================================
    # 2. GLOBAL AVERAGE PLOTS (4 Figures)
    # ===============================================
    plot_global_average_bar(dacapo_df, "Global Average EDP (All E-Core Policies) - DACAPO",
                            out_path / "global_avg_ecore_dacapo_edp.png", e_core_all_policies,
                            baseline_policy='Proactive_E_Oracle', y_limits=None)

    plot_global_average_bar(spec_df, "Global Average EDP (All E-Core Policies) - SPEC",
                            out_path / "global_avg_ecore_spec_edp.png", e_core_all_policies,
                            baseline_policy='Proactive_E_Oracle', y_limits=(0.8, 1.2))

    plot_global_average_bar(dacapo_df, "Global Average EDP (All P-Core Policies) - DACAPO",
                            out_path / "global_avg_pcore_dacapo_edp.png", p_core_all_policies,
                            baseline_policy='Proactive_P_Oracle', y_limits=None)

    plot_global_average_bar(spec_df, "Global Average EDP (All P-Core Policies) - SPEC",
                            out_path / "global_avg_pcore_spec_edp.png", p_core_all_policies,
                            baseline_policy='Proactive_P_Oracle', y_limits=None)

if __name__ == "__main__":
    main()
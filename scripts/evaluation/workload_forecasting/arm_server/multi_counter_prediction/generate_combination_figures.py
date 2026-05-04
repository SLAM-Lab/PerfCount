import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

def main():
    # Configuration
    input_file = 'condensed_pairs_10M.csv'
    output_base_dir = 'grouped_bar_figures_capped'
    
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    # Filter for successful runs and drop missing metrics
    df = df[df['status'] == 'Success'].dropna(subset=['r2_score', 'mape'])
    
    # Clean workload names to extract the "5XX" SPEC name and Phase
    def clean_workload_name(w):
        w_str = str(w)
        match = re.search(r'(5\d{2}\.[a-zA-Z0-9_]+)', w_str)
        if match:
            bench = match.group(1)
        else:
            bench = w_str.split('.')[1].split('_1.0GHz')[0].split('_2.0GHz')[0].split('_3.0GHz')[0] if '.' in w_str else w_str
            
        phase = w_str.split('_')[-1] if 'phase' in w_str else ''
        return f"{bench} ({phase})" if phase else bench
        
    df['workload_short'] = df['workload'].apply(clean_workload_name)
    
    os.makedirs(output_base_dir, exist_ok=True)
    
    group_cols = ['freq', 'horizon', 'timestep']
    grouped = df.groupby(group_cols)
    
    total_groups = len(grouped)
    print(f"Found {total_groups} unique combinations. Generating capped grouped bar charts...")
    
    all_plotted_data = []
    count = 0
    
    for (freq, horizon, timestep), group_data in grouped:
        
        # 1. Find the Top 5 pairs overall for this specific combination
        non_baseline_data = group_data[group_data['pair_name'] != 'Baseline']
        top_5_pairs = non_baseline_data.groupby('pair_name')['r2_score'].mean().nlargest(5).index.tolist()
        
        # 2. Filter the data to ONLY include these top 5 pairs + Baseline
        target_pairs = top_5_pairs + ['Baseline']
        plot_data = group_data[group_data['pair_name'].isin(target_pairs)].copy()
        
        # 3. Calculate the average for these 6 pairs across all workloads
        avg_data = plot_data.groupby('pair_name', observed=False)[['r2_score', 'mape']].mean().reset_index()
        avg_data['workload_short'] = 'AVERAGE'
        avg_data['freq'] = freq
        avg_data['horizon'] = horizon
        avg_data['timestep'] = timestep
        
        plot_data = pd.concat([plot_data, avg_data], ignore_index=True)
        all_plotted_data.append(plot_data[['freq', 'horizon', 'timestep', 'workload_short', 'pair_name', 'r2_score', 'mape']])
        
        # 4. Enforce Categorical orderings
        plot_data['pair_name'] = pd.Categorical(plot_data['pair_name'], categories=['Baseline'] + top_5_pairs, ordered=True)
        
        workload_order = [w for w in sorted(plot_data['workload_short'].unique()) if w != 'AVERAGE'] + ['AVERAGE']
        plot_data['workload_short'] = pd.Categorical(plot_data['workload_short'], categories=workload_order, ordered=True)
        
        plot_data = plot_data.sort_values(['workload_short', 'pair_name'])
        
        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=(24, 16), sharex=True)
        fig.suptitle(f"Top 5 Features vs Baseline across Workloads\nFreq: {freq}GHz | Horizon: {horizon} | Timestep: {timestep}", fontsize=18, y=0.96)
        
        palette = ['#808080', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        # --- Top Plot: R^2 Score ---
        sns.barplot(data=plot_data, x='workload_short', y='r2_score', hue='pair_name', ax=axes[0], palette=palette)
        
        axes[0].set_title('R² Score (Higher is Better)', fontsize=14)
        axes[0].set_ylabel('R² Score', fontsize=12)
        axes[0].axhline(0, color='black', linestyle='-', linewidth=1.0)
        axes[0].grid(True, axis='y', linestyle='--', alpha=0.7)
        axes[0].legend(bbox_to_anchor=(1.01, 1), loc='upper left', title='Feature Pair\n(Ordered by Best Avg)', fontsize=11)
        
        # Cap R^2 axis and label underflows
        axes[0].set_ylim(-0.1, 1.0)
        for container in axes[0].containers:
            for bar, val in zip(container, container.datavalues):
                if pd.notna(val) and val < -0.1:
                    # Place label slightly above the bottom boundary (-0.09) so it's visible
                    axes[0].text(bar.get_x() + bar.get_width()/2, -0.09, 
                                 f'{val:.2f}', ha='center', va='bottom', 
                                 fontsize=10, color='darkred', rotation=90, fontweight='bold')

        # --- Bottom Plot: MAPE ---
        sns.barplot(data=plot_data, x='workload_short', y='mape', hue='pair_name', ax=axes[1], palette=palette)
        
        axes[1].set_title('MAPE (Lower is Better)', fontsize=14)
        axes[1].set_ylabel('MAPE (%)', fontsize=12)
        axes[1].set_xlabel('Benchmark (Sorted by 5XX Name)', fontsize=14)
        axes[1].grid(True, axis='y', linestyle='--', alpha=0.7)
        
        if axes[1].get_legend():
            axes[1].get_legend().remove()
            
        # Cap MAPE axis and label overflows
        axes[1].set_ylim(0, 20)
        for container in axes[1].containers:
            for bar, val in zip(container, container.datavalues):
                if pd.notna(val) and val > 20.0:
                    # Place label slightly below the top boundary (19.5) so it's visible
                    axes[1].text(bar.get_x() + bar.get_width()/2, 19.5, 
                                 f'{val:.1f}', ha='center', va='top', 
                                 fontsize=10, color='darkred', rotation=90, fontweight='bold')
            
        # Rotate X-axis labels
        plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=90, ha="center")
        plt.tight_layout(rect=[0, 0, 0.82, 0.96]) 
        
        # Save figure
        filename = f"CappedBars_Freq{freq}GHz_H{horizon}_T{timestep}.png"
        filepath = os.path.join(output_base_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig) 
        
        count += 1
        if count % 10 == 0:
            print(f"Processed {count}/{total_groups} combinations...")

    print(f"\nAll capped grouped bar charts saved to '{output_base_dir}'.")
    
    # ---------------------------------------------------------
    # DUMP THE PLOTTED DATA TO CSVS
    # ---------------------------------------------------------
    print("Generating CSV data dumps...")
    master_plot_df = pd.concat(all_plotted_data, ignore_index=True)
    
    r2_pivot = master_plot_df.pivot_table(
        index=['freq', 'horizon', 'timestep', 'workload_short'], 
        columns='pair_name', values='r2_score', aggfunc='first'
    ).reset_index()
    
    mape_pivot = master_plot_df.pivot_table(
        index=['freq', 'horizon', 'timestep', 'workload_short'], 
        columns='pair_name', values='mape', aggfunc='first'
    ).reset_index()
    
    r2_pivot.to_csv('plotted_data_r2.csv', index=False)
    mape_pivot.to_csv('plotted_data_mape.csv', index=False)
    print("Successfully saved 'plotted_data_r2.csv' and 'plotted_data_mape.csv'.")

if __name__ == "__main__":
    sns.set_theme(style="whitegrid")
    main()

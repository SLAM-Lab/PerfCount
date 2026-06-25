import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================
# 1. CONFIGURATION
# =========================================================
MACHINE = "arm_server"
FREQS = ["1.0", "2.0", "3.0"]
CORRELATION_THRESHOLD = 0.90  # Flag pairs with correlation > 90%

ALL_COUNTERS = "cpu_cycles branch_misses branches bus_access cache_misses cache_references dtlb_load_misses dtlb_loads instructions itlb-load-misses itlb-loads l1-dcache-load-misses l1-dcache-loads l1_icache_load_misses l1_icache_loads l1d_cache l1d_cache_refill l1d_cache_wb l1i_cache l1i_cache_refill l2d_cache l2d_cache_refill l2d_cache_wb mem_access stalled_cycles_backend stalled_cycles_frontend".split()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, f"../../../../processed_data/{MACHINE}"))

def main():
    for freq in FREQS:
        print(f"\n========================================")
        print(f"[*] Analyzing Cross-Correlation for {freq} GHz")
        print(f"========================================")
        
        search_pattern = os.path.join(DATA_DIR, "**", f"aligned_*_{freq}GHz_phase*.csv")
        csv_files = glob.glob(search_pattern, recursive=True)
        
        if not csv_files:
            print(f"[-] Could not find raw data files at {search_pattern}. Skipping...")
            continue

        print(f"[*] Found {len(csv_files)} workload files. Loading data...")
        
        # Load and concatenate a sample from each workload to build a global profile
        df_list = []
        for file in csv_files:
            try:
                # Taking every 10th row saves memory and time while preserving the statistical distribution
                temp_df = pd.read_csv(file, usecols=ALL_COUNTERS)
                temp_df = temp_df.iloc[::10, :] 
                df_list.append(temp_df)
            except Exception as e:
                print(f"Skipping {file}: {e}")
                
        if not df_list:
            continue

        # Combine all data
        global_df = pd.concat(df_list, ignore_index=True)
        print(f"[*] Built global dataset with {len(global_df)} samples. Computing correlation matrix...")

        # Compute Pearson Correlation
        corr_matrix = global_df.corr()

        # =========================================================
        # 2. PLOT THE HEATMAP
        # =========================================================
        plt.figure(figsize=(20, 16))
        
        # Generate a mask for the upper triangle so we don't see duplicate mirrored squares
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Custom diverging colormap
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        
        sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1.0, vmin=-1.0, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .7}, annot=False)
        
        plt.title(f"Hardware Counter Cross-Correlation Heatmap ({freq} GHz)", fontsize=20, pad=20)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(fontsize=10)
        plt.tight_layout()
        
        heatmap_file = f'counter_correlation_heatmap_{freq}GHz.png'
        plt.savefig(heatmap_file, dpi=300)
        plt.close()
        print(f"[+] Saved heatmap to '{heatmap_file}'")

        # =========================================================
        # 3. FIND REDUNDANT PAIRS
        # =========================================================
        redundant_pairs = []
        
        # Iterate through the upper triangle of the matrix to find high correlations
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > CORRELATION_THRESHOLD:
                    redundant_pairs.append({
                        'Counter 1': corr_matrix.columns[i],
                        'Counter 2': corr_matrix.columns[j],
                        'Correlation': corr_value
                    })
                    
        redundant_df = pd.DataFrame(redundant_pairs)
        if not redundant_df.empty:
            redundant_df = redundant_df.sort_values(by='Correlation', key=abs, ascending=False)
        
        csv_out = f'redundant_counters_{freq}GHz.csv'
        redundant_df.to_csv(csv_out, index=False)
        print(f"[+] Found {len(redundant_df)} highly correlated pairs (> {CORRELATION_THRESHOLD}). Saved to '{csv_out}'")
        
        # Print the top 5 to the console for a quick peek
        if len(redundant_df) > 0:
            print(f"\n--- Top Redundant Counter Pairs ({freq} GHz) ---")
            print(redundant_df.head(5).to_string(index=False))

if __name__ == '__main__':
    main()
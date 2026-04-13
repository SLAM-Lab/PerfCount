import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

def generate_phase_plots(wl, ph, m_type, all_traces, min_len, bar_dir, trace_dir, category):
    
    # --- 1. STANDARDIZE ORDER ---
    # Sort keys alphabetically, but force the Oracle baseline to the front
    oracle_key = 'Proactive_Hetero_Oracle'
    sorted_keys = sorted(list(all_traces.keys()))
    if oracle_key in sorted_keys:
        sorted_keys.remove(oracle_key)
        sorted_keys.insert(0, oracle_key)

    # Use matplotlib's tab20 colormap which has 20 distinct colors. 
    cmap = plt.get_cmap('tab20')
    num_traces = len(sorted_keys)
    colors = [cmap(i % 20) for i in range(num_traces)]
    
    # Extract finals using our standardized, sorted order
    finals = {k: all_traces[k][-1] for k in sorted_keys}
    
    # Safely get the baseline value
    base_val = finals.get(oracle_key, min(finals.values()) + 1e-12)
    
    # --- 2. CSV DIRECTORY SETUP ---
    # Create a csv_data folder at the same level as bar_dir/trace_dir
    csv_dir = bar_dir.parent / "csv_data"
    csv_dir.mkdir(parents=True, exist_ok=True)

    # --- 3. BAR PLOT ---
    plt.figure(figsize=(14, 7))  
    norm_vals = [finals[k] / base_val for k in sorted_keys]
    
    # Export Bar Data to CSV
    bar_df = pd.DataFrame({
        'Policy': sorted_keys,
        f'Final_{m_type}': [finals[k] for k in sorted_keys],
        f'Normalized_{m_type}': norm_vals
    })
    bar_df.to_csv(csv_dir / f"bar_{category}_{m_type}_{wl}_phase{ph}.csv", index=False)

    # Make x-axis labels wrap slightly better for the bar chart
    display_keys = [k.replace('_', '\n') for k in sorted_keys]
    
    plt.bar(display_keys, norm_vals, color=colors, edgecolor='black')
    plt.axhline(1.0, color='red', linestyle='--', linewidth=2, label='Global Oracle Baseline')
    
    plt.title(f"{wl} Phase {ph} ({category}) - {m_type}\nNormalized to Proactive Hetero Oracle")
    plt.ylabel(f"Relative {m_type} Score")
    plt.xticks(rotation=45, ha='right') # Angle text to avoid overlaps
    plt.tight_layout()
    
    plt.savefig(bar_dir / f"bar_{category}_{m_type}_{wl}_phase{ph}.png", dpi=100)
    plt.close()
    
    # --- 4. TRACE PLOT ---
    plt.figure(figsize=(14, 8)) 
    x_axis = np.arange(min_len)
    
    # Initialize a Python dictionary to hold the trace data (prevents Pandas fragmentation warnings)
    trace_data_dict = {'Time_Chunk': x_axis}
    
    for idx, name in enumerate(sorted_keys):
        trace = all_traces[name]
        gap_trace = (trace / (all_traces.get(oracle_key, trace) + 1e-12)) - 1.0
        gap_trace = np.clip(gap_trace, 0, 5.0) # Clip at +500% penalty to keep readable
        
        # Populate the dictionary instead of modifying a DataFrame in a loop
        trace_data_dict[f'{name}_Raw'] = trace
        trace_data_dict[f'{name}_Gap_Penalty_%'] = gap_trace * 100
        
        style = '-' if 'Oracle' in name else '--'
        plt.plot(x_axis, gap_trace * 100, label=name.replace('_Hetero', '').replace('_DVFS', ''), 
                 color=colors[idx], linestyle=style, linewidth=2)
        
    # Convert dictionary to DataFrame all at once, avoiding the fragmentation warning
    trace_df = pd.DataFrame(trace_data_dict)
    
    # Export Trace Data to CSV
    trace_df.to_csv(csv_dir / f"trace_{category}_{m_type}_{wl}_phase{ph}.csv", index=False)
        
    plt.title(f"{wl} Phase {ph} ({category}) - {m_type} Gap Over Time\n(% Worse than Global Oracle)")
    plt.xlabel("Execution Progress (10M Instruction Chunks)")
    plt.ylabel(f"% Penalty vs Global Oracle")
    
    # Push legend outside the plot, and use 2 columns if there are lots of policies
    ncol = 2 if num_traces > 10 else 1
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', ncol=ncol)
    
    # Use bbox_inches='tight' so the saved image doesn't cut off the external legend
    plt.savefig(trace_dir / f"trace_{category}_{m_type}_{wl}_phase{ph}.png", dpi=100, bbox_inches='tight')
    plt.close()
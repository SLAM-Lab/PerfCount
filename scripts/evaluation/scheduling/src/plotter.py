import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def generate_phase_plots(wl, ph, m_type, all_traces, min_len, bar_dir, trace_dir, category):
    # Use matplotlib's tab20 colormap which has 20 distinct colors. 
    # The modulo (%) ensures it safely wraps around if we ever exceed 20 policies!
    cmap = plt.get_cmap('tab20')
    num_traces = len(all_traces)
    colors = [cmap(i % 20) for i in range(num_traces)]
    
    finals = {k: v[-1] for k, v in all_traces.items()}
    oracle_key = 'Proactive_Hetero_Oracle'
    
    # Safely get the baseline value
    base_val = finals.get(oracle_key, min(finals.values()) + 1e-12)
    
    # --- 1. BAR PLOT ---
    plt.figure(figsize=(14, 7))  # Made slightly wider to fit 15 bars
    norm_vals = [finals[k] / base_val for k in finals.keys()]
    
    plt.bar(finals.keys(), norm_vals, color=colors, edgecolor='black')
    plt.axhline(1.0, color='red', linestyle='--', linewidth=2, label='Global Oracle Baseline')
    
    plt.title(f"{wl} Phase {ph} ({category}) - {m_type}\nNormalized to Proactive Hetero Oracle")
    plt.ylabel(f"Relative {m_type} Score")
    
    # Rotate labels by 45 degrees so 15 policies don't overlap
    plt.xticks(rotation=45, ha='right')
    
    # Cap Y limit intelligently so one bad policy doesn't squash the rest
    max_val = max(norm_vals) if norm_vals else 1.0
    plt.ylim(0.9, max(1.05, min(max_val * 1.05, 5.0))) 
    plt.tight_layout()
    plt.savefig(bar_dir / f"bar_{category}_{m_type}_{wl}_phase{ph}.png", dpi=100)
    plt.close()
    
    # --- 2. TRACE PLOT ---
    plt.figure(figsize=(14, 8)) # Taller to accommodate a larger legend
    x_axis = np.arange(min_len)
    
    for idx, (name, trace) in enumerate(all_traces.items()):
        gap_trace = (trace / (all_traces.get(oracle_key, trace) + 1e-12)) - 1.0
        gap_trace = np.clip(gap_trace, 0, 5.0) # Clip at +500% penalty to keep readable
        
        style = '-' if 'Oracle' in name else '--'
        plt.plot(x_axis, gap_trace * 100, label=name.replace('_Hetero', '').replace('_DVFS', ''), 
                 color=colors[idx], linestyle=style, linewidth=2)
        
    plt.title(f"{wl} Phase {ph} ({category}) - {m_type} Gap Over Time\n(% Worse than Global Oracle)")
    plt.xlabel("Execution Progress (10M Instruction Chunks)")
    plt.ylabel(f"% Penalty vs Global Oracle")
    
    # Push legend outside the plot, and use 2 columns if there are lots of policies
    ncol = 2 if num_traces > 10 else 1
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', ncol=ncol)
    
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.savefig(trace_dir / f"trace_{category}_{m_type}_{wl}_phase{ph}.png", dpi=100)
    plt.close()

def save_master_csv(master_results, output_path):
    df = pd.DataFrame(master_results)
    csv_path = output_path / "scheduler_modular_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"Master CSV summary saved to {csv_path}")
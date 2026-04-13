import argparse
import os
import re
import pandas as pd
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Generate workload-phase specific speedup files and a condensed average summary.")
    parser.add_argument('--input_dir', type=str, required=True, help="Path to the directory containing the trace CSVs")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory where the result CSVs will be written")
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    
    if not input_path.exists() or not input_path.is_dir():
        print(f"Error: Input directory {args.input_dir} does not exist.")
        return

    # Create output directory and a subfolder for the granular files
    output_path.mkdir(parents=True, exist_ok=True)
    granular_dir = output_path / "granular_phase_traces"
    granular_dir.mkdir(exist_ok=True)

    # Regex to extract metadata
    pattern = re.compile(r"aligned_spec_(.+)_([0-9.]+)GHz_cpu(\d+)_phase(\d+)\.csv")
    
    all_data = []
    files_processed = 0

    print(f"Scanning directory: {input_path}")
    
    # 1. Read files and extract variables
    for f in input_path.glob("aligned_spec_*.csv"):
        match = pattern.match(f.name)
        if not match: continue
            
        workload = match.group(1)
        freq_ghz = float(match.group(2))
        cpu_id = match.group(3)
        phase = match.group(4) 
        
        core_type = 'P' if cpu_id == '0' else 'E'
        config_name = f"{core_type}_{freq_ghz}GHz"
        
        try:
            # FIX 1: Tell Pandas to load the ref_cycles column instead of cpu_cycles
            df = pd.read_csv(f, usecols=['sample_index', 'ref_cycles'])
            
            # FIX 2: Calculate generic time using ref_cycles (which ticks at a constant hardware frequency)
            df['time_s'] = df['ref_cycles'] / 1e9  
            
            df['workload'] = workload
            df['phase'] = phase
            df['config'] = config_name
            
            all_data.append(df[['workload', 'phase', 'sample_index', 'config', 'time_s']])
            files_processed += 1
        except Exception as e:
            print(f"Error reading {f.name}: {e}")

    if not all_data:
        print("No valid CSV files found.")
        return
        
    print(f"Successfully loaded {files_processed} files. Condensing data...")

    # 2. Master Pivot
    master_df = pd.concat(all_data, ignore_index=True)
    pivot_df = master_df.pivot_table(
        index=['workload', 'phase', 'sample_index'], 
        columns='config', 
        values='time_s'
    ).reset_index()
    
    configs = [c for c in pivot_df.columns if c not in ['workload', 'phase', 'sample_index']]
    
    print(f"Found {len(configs)} unique processor/frequency configurations.")
    print(f"Truncating alignments and generating granular CSVs in {granular_dir}...")

    summary_records = []

    # 3. Iterate over every Source Configuration
    for source_config in configs:
        
        # 4. Group by specific Workload and Phase
        for (wl, ph), group_df in pivot_df.groupby(['workload', 'phase']):
            
            active_configs = [c for c in configs if not group_df[c].isna().all()]
            
            if source_config not in active_configs:
                continue
            
            group_clean = group_df.dropna(subset=active_configs).copy()
            group_clean = group_clean[(group_clean[active_configs] > 0).all(axis=1)]
            
            if group_clean.empty:
                continue
                
            res_df = group_clean[['sample_index']].copy()
            res_df[f'Time_{source_config}'] = group_clean[source_config]
            
            # 5. Calculate target speedups for this phase
            for target_config in active_configs:
                if target_config == source_config:
                    continue
                    
                col_name = f"Speedup_{target_config}_vs_{source_config}"
                res_df[col_name] = group_clean[source_config] / group_clean[target_config]
                
                avg_speedup = res_df[col_name].mean()
                summary_records.append({
                    'Source_Config': source_config,
                    'Target_Config': target_config,
                    'Workload': wl,
                    'Phase': ph,
                    'Avg_Time_Source_s': group_clean[source_config].mean(),
                    'Avg_Time_Target_s': group_clean[target_config].mean(),
                    'Avg_Speedup': avg_speedup
                })
            
            # Save granular file
            file_name = f"speedups_{source_config}_{wl}_phase{ph}.csv"
            out_file = granular_dir / file_name
            res_df.to_csv(out_file, index=False)

    print(f"Finished generating granular phase files!")

    # 6. Save the Condensed Master Summary
    summary_df = pd.DataFrame(summary_records)
    summary_df = summary_df.sort_values(by=['Source_Config', 'Workload', 'Phase', 'Avg_Speedup'], ascending=[True, True, True, False])
    
    master_out = output_path / "condensed_average_speedups_summary.csv"
    summary_df.to_csv(master_out, index=False)
    
    print(f"Success! Condensed summary saved to: {master_out.name}")

if __name__ == "__main__":
    main()
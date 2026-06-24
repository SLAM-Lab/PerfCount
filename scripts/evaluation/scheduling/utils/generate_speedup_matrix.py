import argparse
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from pathlib import Path


def process_group(wl, ph, group_df, configs, granular_dir):
    """Handle one (workload, phase) group: compute the shared group_clean
    once, then write a granular speedups_<config>_<wl>_phase<ph>.csv for
    every active source config. Returns the summary records for this group."""
    summary_records = []

    active_configs = [c for c in configs if not group_df[c].isna().all()]
    if not active_configs:
        return summary_records

    group_clean = group_df.dropna(subset=active_configs).copy()
    group_clean = group_clean[(group_clean[active_configs] > 0).all(axis=1)]
    if group_clean.empty:
        return summary_records

    for source_config in active_configs:
        res_df = group_clean[['sample_index']].copy()
        res_df[f'Time_{source_config}'] = group_clean[source_config]

        # Direct (non-speedup-derived) measured power per config, for any
        # active config that has measured power data for this phase.
        for cfg in active_configs:
            power_col = f'Power_{cfg}'
            if power_col in group_clean.columns and not group_clean[power_col].isna().all():
                res_df[power_col] = group_clean[power_col]

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

        file_name = f"speedups_{source_config}_{wl}_phase{ph}.csv"
        res_df.to_csv(granular_dir / file_name, index=False)

    return summary_records

def main():
    parser = argparse.ArgumentParser(description="Generate workload-phase specific speedup files and a condensed average summary.")
    parser.add_argument('--input_dir', type=str, required=True, help="Path to the directory containing the trace CSVs")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory where the result CSVs will be written")
    parser.add_argument('--arch', type=str, default='x86', choices=['x86', 'arm_edge'],
                         help="Target architecture: 'x86' (P/E cores) or 'arm_edge' (L/B cores)")
    parser.add_argument('--workers', type=int, default=1,
                         help="number of (workload, phase) groups to process in parallel (default 1)")
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

    # FIX 1: Updated Regex to capture anything after 'aligned_' as the workload name
    pattern = re.compile(r"aligned_(.+)_([0-9.]+)GHz_cpu(\d+)_phase(\d+)\.csv")
    
    all_data = []
    files_processed = 0

    print(f"Scanning directory: {input_path}")
    
    # FIX 2: Updated the glob to scan for all 'aligned_' files
    for f in input_path.glob("aligned_*.csv"):
        match = pattern.match(f.name)
        if not match: continue
            
        workload = match.group(1) # This will now be 'spec_500.perlbench_r' or 'dacapo_avrora'
        freq_ghz = float(match.group(2))
        cpu_id = match.group(3)
        phase = match.group(4) 
        
        if args.arch == 'arm_edge':
            core_type = 'L' if int(cpu_id) < 4 else 'B'
        else:
            core_type = 'P' if cpu_id == '0' else 'E'
        config_name = f"{core_type}_{freq_ghz}GHz"

        try:
            usecols = ['sample_index', 'ref_cycles']
            has_power = 'power_watts_total_block' in pd.read_csv(f, nrows=0).columns
            if has_power:
                usecols.append('power_watts_total_block')
            df = pd.read_csv(f, usecols=usecols)

            ref_divisor = 2e9 if args.arch == 'arm_edge' else 1e9
            df['time_s'] = df['ref_cycles'] / ref_divisor

            df['workload'] = workload
            df['phase'] = phase
            df['config'] = config_name
            df['power_w'] = df['power_watts_total_block'] if has_power else np.nan

            all_data.append(df[['workload', 'phase', 'sample_index', 'config', 'time_s', 'power_w']])
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

    # Measured per-chunk power, pivoted the same way and joined in as Power_<config>
    # columns. Configs with no measured power (e.g. P_5.0GHz placeholder) end up
    # all-NaN here and are simply not written to the granular files.
    power_pivot = master_df.pivot_table(
        index=['workload', 'phase', 'sample_index'],
        columns='config',
        values='power_w'
    )
    power_pivot.columns = [f'Power_{c}' for c in power_pivot.columns]
    power_pivot = power_pivot.reset_index()
    pivot_df = pivot_df.merge(power_pivot, on=['workload', 'phase', 'sample_index'], how='left')
    
    print(f"Found {len(configs)} unique processor/frequency configurations.")
    print(f"Truncating alignments and generating granular CSVs in {granular_dir}...")

    summary_records = []

    # 3. Process each (workload, phase) group independently -- group_clean is
    # computed once per group and reused across all source configs.
    groups = list(pivot_df.groupby(['workload', 'phase']))

    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(process_group, wl, ph, group_df, configs, granular_dir)
                for (wl, ph), group_df in groups
            ]
            for future in as_completed(futures):
                summary_records.extend(future.result())
    else:
        for (wl, ph), group_df in groups:
            summary_records.extend(process_group(wl, ph, group_df, configs, granular_dir))

    print(f"Finished generating granular phase files!")

    # 6. Save the Condensed Master Summary
    summary_df = pd.DataFrame(summary_records)
    summary_df = summary_df.sort_values(by=['Source_Config', 'Workload', 'Phase', 'Avg_Speedup'], ascending=[True, True, True, False])
    
    master_out = output_path / "condensed_average_speedups_summary.csv"
    summary_df.to_csv(master_out, index=False)
    
    print(f"Success! Condensed summary saved to: {master_out.name}")

if __name__ == "__main__":
    main()
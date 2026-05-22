import os
import glob
import pandas as pd
import numpy as np
import re
import subprocess
import argparse
import multiprocessing as mp

def parse_filename(filename):
    match = re.search(r"cpu_(?P<cpu_id>\d+)_(?P<freq>[\d\.]+)GHz_(?P<bench>dacapo_.+)_10000000_(?P<run>\d+)_(?P<phase>\d+)\.out", filename)
    if match:
        return match.groupdict()
    return None

def clean_event_name(raw_event):
    name = raw_event.rstrip(':')
    name = name.split(':')[0]
    
    # Handle ARM specific PMU prefixes like armv8_pmuv3_0/br_pred/
    if 'armv8_pmuv3' in name and '/' in name:
        parts = name.split('/')
        if len(parts) >= 2:
            name = parts[1]
    elif '/' in name:
        parts = name.split('/')
        if len(parts) >= 2 and parts[1]:
            name = parts[1] 
        else:
            name = parts[0]

    name = name.lower()

    mapping = {
        'cpu-cycles': 'cpu_cycles',
        'cycles': 'cpu_cycles',
        'instructions': 'instructions',
        
        # Branch Prediction (Group 1)
        'br_pred': 'branches', # Normalized to 'branches' for baseline comparison
        'br_mis_pred': 'branch_misses',
        
        # L1 Data Cache (Groups 1, 2, 3)
        'l1-dcache-loads': 'l1_dcache_loads',
        'l1-dcache-load-misses': 'l1_dcache_load_misses',
        'l1d_cache': 'l1d_cache',
        'l1d_cache_refill': 'l1d_cache_refill',
        'l1d_cache_wb': 'l1d_cache_wb',
        
        # L1 Instruction Cache (Group 3, 4)
        'l1-icache-loads': 'l1_icache_loads',
        'l1-icache-load-misses': 'l1_icache_load_misses',
        'l1i_cache': 'l1i_cache',
        'l1i_cache_refill': 'l1i_cache_refill',
        
        # L2 Data Cache (Group 4, 5)
        'l2d_cache': 'l2d_cache',
        'l2d_cache_refill': 'l2d_cache_refill',
        'l2d_cache_wb': 'l2d_cache_wb',
        'cache-references': 'cache_references',
        'cache-misses': 'cache_misses',
        
        # TLB Metrics (Group 6, 7)
        'dtlb-loads': 'dtlb_loads',
        'dtlb-load-misses': 'dtlb_load_misses',
        'itlb-loads': 'itlb_loads',
        'itlb-load-misses': 'itlb_load_misses',
        
        # System & Memory Metrics (Group 0, 7, 8)
        'stalled-cycles-backend': 'stalled_cycles_backend',
        'stalled-cycles-frontend': 'stalled_cycles_frontend',
        'bus_access': 'bus_access',
        'mem_access': 'mem_access',
        'memory_error': 'memory_error',
        'exc_return': 'exc_return'
    }

    if name in mapping:
        return mapping[name]

    # Fallback for any other events
    return name.replace('-', '_').replace('.', '_')

def merge_split_blocks(df, target_instructions=10000000):
    if df.empty or 'instructions' not in df.columns:
        return df
        
    merged_rows = []
    current_row = None
    threshold = target_instructions * 0.98 
    
    for _, row in df.iterrows():
        if current_row is None:
            current_row = row.copy()
        else:
            for col in df.columns:
                if col != 'sample_index':
                    current_row[col] += row[col]
        
        if current_row['instructions'] >= threshold:
            merged_rows.append(current_row)
            current_row = None
            
    if current_row is not None:
        merged_rows.append(current_row)
        
    merged_df = pd.DataFrame(merged_rows).reset_index(drop=True)
    merged_df['sample_index'] = range(len(merged_df))
    return merged_df

def repair_dropped_samples(df, target=10000000):
    if df.empty or 'instructions' not in df.columns:
        return df
        
    repaired_rows = []
    for _, row in df.iterrows():
        instrs = row.get('instructions', 0)
        
        if pd.isna(instrs) or instrs == 0:
            repaired_rows.append(row)
            continue
            
        ratio = instrs / target
        if ratio >= 1.85:
            splits = int(round(ratio))
            split_row = row.copy()
            for col in df.columns:
                if col != 'sample_index' and pd.api.types.is_numeric_dtype(df[col]):
                    split_row[col] = int(round(split_row[col] / splits))
            for _ in range(splits):
                repaired_rows.append(split_row.copy())
        else:
            repaired_rows.append(row)
            
    repaired_df = pd.DataFrame(repaired_rows).reset_index(drop=True)
    if 'sample_index' in repaired_df.columns:
        repaired_df['sample_index'] = range(len(repaired_df))
    return repaired_df

def check_block_variance(df, fname, target=10000000):
    messages = []
    if df.empty or 'instructions' not in df.columns:
        return messages
    
    instrs = df['instructions']
    
    if len(instrs) > 1 and instrs.iloc[-1] < (target * 0.90):
        instrs_to_check = instrs.iloc[:-1]
    else:
        instrs_to_check = instrs

    mean_val = instrs_to_check.mean()
    std_val = instrs_to_check.std()
    min_val = instrs_to_check.min()
    max_val = instrs_to_check.max()
    
    lower_bound = target * 0.95
    upper_bound = target * 1.05
    
    outliers = instrs_to_check[(instrs_to_check < lower_bound) | (instrs_to_check > upper_bound)]
    
    if not outliers.empty:
        messages.append(f"  [CHECK] {fname} variance:")
        messages.append(f"      Mean: {mean_val:,.0f} | Std: {std_val:,.0f} | Min: {min_val:,.0f} | Max: {max_val:,.0f}")
        messages.append(f"      [!] Found {len(outliers)} significant outliers (> 5% deviation from {target:,}):")
        for idx, val in outliers.items():
            messages.append(f"          Row {idx}: {val:,.0f} instructions")
            
    return messages

def parse_perf_script_output(proc_stdout, arch="arm"):
    data = []
    current_interval = {}
    
    for line in proc_stdout:
        line = line.decode('utf-8', errors='replace').strip()
        if not line or line.startswith('#'):
            continue
        
        parts = line.split()
        if len(parts) < 3:
            continue

        # Skip C2 JIT compiler thread samples
        if parts[0] == 'C2':
            continue

        ts_idx = -1
        for i, p in enumerate(parts):
            if p.endswith(':'):
                try:
                    float(p[:-1])
                    ts_idx = i
                    break
                except ValueError:
                    continue
        
        if ts_idx == -1 or ts_idx + 2 >= len(parts):
            continue

        ts_str = parts[ts_idx].replace(':', '')
        val_str = parts[ts_idx + 1].replace(',', '')
        raw_event = parts[ts_idx + 2]

        try:
            value = int(val_str)
        except ValueError:
            if val_str == '<not':
                value = 0
            else:
                continue

        event_name = clean_event_name(raw_event)

        if 'ts' in current_interval and current_interval['ts'] != ts_str:
            data.append(current_interval)
            current_interval = {}
        
        current_interval['ts'] = ts_str
        current_interval[event_name] = value
        
    if current_interval:
        data.append(current_interval)
        
    df = pd.DataFrame(data)
    if not df.empty:
        if 'ts' in df.columns:
            df.drop(columns=['ts'], inplace=True)
        if arch == "x86":
            df = merge_split_blocks(df)
        df = repair_dropped_samples(df)
        if 'sample_index' not in df.columns:
            df['sample_index'] = range(len(df))
            
    return df

# Helper to unwrap arguments for imap_unordered
def process_single_file_wrapper(args):
    f, out_dir, arch = args
    fname = os.path.basename(f)
    meta = parse_filename(fname)
    
    if not meta:
        return False, f"Skipped {fname} (doesn't match Dacapo naming convention)", []
        
    try:
        cmd = ["perf", "script", "-i", f]
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
            df = parse_perf_script_output(proc.stdout, arch=arch)
            
        if df.empty:
            return False, f"[WARN] {fname}: No data extracted.", []

        outlier_messages = check_block_variance(df, fname, target=10000000)

        out_name = f"{meta['bench']}_{meta['freq']}GHz_cpu{meta['cpu_id']}_run{meta['run']}_phase{meta['phase']}.csv"
        out_path = os.path.join(out_dir, out_name)
        
        df.to_csv(out_path, index=False)
        return True, f"Processed {fname}", outlier_messages
        
    except Exception as e:
        return False, f"[ERR] Failed {fname}: {e}", []

def align_csvs_and_evaluate(out_dir):
    print("\n--- Starting Dacapo Alignment & Evaluation Phase ---")
    csv_files = glob.glob(os.path.join(out_dir, "dacapo_*.csv"))
    csv_files = [f for f in csv_files if not os.path.basename(f).startswith("aligned_")]
    
    if not csv_files:
        print("No CSVs found to align.")
        return

    groups = {}
    for f in csv_files:
        fname = os.path.basename(f)
        match = re.search(r"(?P<bench>dacapo_.+)_(?P<freq>[\d\.]+)GHz_cpu(?P<cpu>\d+)_run(?P<run>\d+)_phase(?P<phase>\d+)\.csv", fname)
        if match:
            key = (match.group('bench'), match.group('freq'), match.group('phase'), match.group('cpu'))
            groups.setdefault(key, []).append((int(match.group('run')), f))

    alignment_scores = []

    for (bench, freq, phase, cpu), files_info in groups.items():
        files_info.sort(key=lambda x: x[0])
        
        aligned_df = None
        seen_columns = set()
        instructions_across_runs = {}

        for run, f in files_info:
            df = pd.read_csv(f)
            
            # Extract instructions for correlation testing BEFORE dropping columns
            if 'instructions' in df.columns:
                instructions_across_runs[f"Run_{run}"] = df['instructions']
            
            if aligned_df is not None:
                cols_to_keep = ['sample_index'] + [c for c in df.columns if c not in seen_columns and c != 'sample_index']
                df = df[cols_to_keep]
            
            seen_columns.update(df.columns)
            
            if aligned_df is None:
                aligned_df = df
            else:
                aligned_df = pd.merge(aligned_df, df, on='sample_index', how='outer')

        if len(instructions_across_runs) > 1:
            inst_df = pd.DataFrame(instructions_across_runs).dropna()
            if not inst_df.empty and len(inst_df) > 5:
                corr_matrix = inst_df.corr(method='spearman')
                triu_indices = np.triu_indices_from(corr_matrix.values, k=1)
                
                if len(triu_indices[0]) > 0:
                    avg_corr = corr_matrix.values[triu_indices].mean()
                    alignment_scores.append({
                        "Benchmark": bench.replace('dacapo_', ''),
                        "Freq": f"{freq} GHz",
                        "CPU": f"Core {cpu}",
                        "Phase": phase,
                        "Runs": len(instructions_across_runs),
                        "Avg_Corr": avg_corr
                    })

        cols = ['sample_index'] + sorted([c for c in aligned_df.columns if c != 'sample_index'])
        aligned_df = aligned_df[cols].fillna(0).astype('int64')
        aligned_df = aligned_df[aligned_df["instructions"] > 0]

        aligned_out = os.path.join(out_dir, f"aligned_{bench}_{freq}GHz_cpu{cpu}_phase{phase}.csv")
        aligned_df.to_csv(aligned_out, index=False)
        print(f"Created aligned trace: {os.path.basename(aligned_out)}")

    if alignment_scores:
        print("\n=======================================================")
        print("             DACAPO ALIGNMENT SUMMARY                  ")
        print("=======================================================")
        score_df = pd.DataFrame(alignment_scores).sort_values(by=["Freq", "Benchmark", "CPU"])
        print(score_df.to_string(index=False))
        
        overall_mean = score_df['Avg_Corr'].mean()
        print("\n=======================================================")
        print(f"OVERALL AVERAGE SPEARMAN CORRELATION: {overall_mean:.4f}")
        print("=======================================================")

def main():
    parser = argparse.ArgumentParser(description="Process raw ARM Dacapo perf .out files into CSVs, stripping C2 JIT compiler samples.")
    parser.add_argument("--raw_dir", default="../../../raw_data/arm_server", help="Directory with raw .out files")
    parser.add_argument("--out_dir", default="../../../processed_data/arm_server_no_c2", help="Directory to save CSVs")
    parser.add_argument("--jobs", type=int, default=40, help="Number of parallel workers")
    parser.add_argument("--arch", choices=["x86", "arm"], default="arm", help="Target architecture (default: arm)")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    files = glob.glob(os.path.join(args.raw_dir, "*dacapo*.out"))
    print(f"Found {len(files)} Dacapo raw files in {args.raw_dir}")
    print(f"Processing to {args.out_dir} using {args.jobs} workers (Arch: {args.arch})...")
    
    # Bundle tasks for the memory-safe multiprocessing map
    tasks = [(f, args.out_dir, args.arch) for f in files]
    count = 0
    skipped = 0
    
    # MAGIC FIX: maxtasksperchild=1 strictly forces process garbage collection
    with mp.Pool(processes=args.jobs, maxtasksperchild=1) as pool:
        for success, msg, outlier_messages in pool.imap_unordered(process_single_file_wrapper, tasks):
            if success:
                count += 1
            else:
                skipped += 1
                print(f"SKIPPED: {msg}")
                
            for out_msg in outlier_messages:
                print(out_msg)

            if count > 0 and count % 20 == 0:
                print(f"  Processed {count} files...")
            
    print(f"\nDone. Processed {count} files. Skipped {skipped}.")
    
    align_csvs_and_evaluate(args.out_dir)

if __name__ == "__main__":
    main()
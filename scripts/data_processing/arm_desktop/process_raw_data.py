import os
import glob
import pandas as pd
import re
import subprocess
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

def parse_filename(filename):
    match = re.search(r"cpu_(?P<cpu_id>\d+)_(?P<freq>[\d\.]+)GHz_(?P<bench>.+)_10000000_(?P<run>\d+)_(?P<phase>\d+)\.out", filename)
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

    # Exact dictionary mapping for ARM Desktop PMU counters
    mapping = {
        'cpu-cycles': 'cpu_cycles',
        'cycles': 'cpu_cycles',
        'instructions': 'instructions',
        
        # Run 0 & 11
        'branch-instructions': 'branches',
        'branches': 'branches',
        'branch-misses': 'branch_misses',
        'branch-loads': 'branch_loads',
        'branch-load-misses': 'branch_load_misses',
        
        # Run 1, 2, 7, 8, 10
        'l1-dcache-loads': 'l1_dcache_loads',
        'l1-dcache-load-misses': 'l1_dcache_load_misses',
        'l1-icache-loads': 'l1_icache_loads',
        'l1-icache-load-misses': 'l1_icache_load_misses',
        'cache-misses': 'cache_misses',
        'cache-references': 'cache_references',
        'l1d_cache': 'l1d_cache',
        'l1i_cache': 'l1i_cache',
        
        # Run 2, 3, 6, 8, 9
        'dtlb-loads': 'dtlb_loads',
        'dtlb-load-misses': 'dtlb_load_misses',
        'itlb-loads': 'itlb_loads',
        'itlb-load-misses': 'itlb_load_misses',
        'dtlb_walk': 'dtlb_walk',
        'itlb_walk': 'itlb_walk',
        
        # Run 3, 12
        'context-switches': 'context_switches',
        'cs': 'context_switches',
        
        # Run 4, 5, 17
        'page-faults': 'page_faults',
        'alignment-faults': 'alignment_faults',
        'emulation-faults': 'emulation_faults',
        'minor-faults': 'minor_faults',
        'major-faults': 'major_faults',
        'faults': 'faults',
        'cpu-migrations': 'cpu_migrations',
        'migrations': 'cpu_migrations',
        'memory_error': 'memory_error',
        
        # Run 10, 12
        'system_time': 'system_time',
        'task-clock': 'task_clock',
        'cpu-clock': 'cpu_clock',
        
        # Run 13, 14, 15
        'bx_stall': 'bx_stall',
        'fx_stall': 'fx_stall',
        'ixa_stall': 'ixa_stall',
        'ixb_stall': 'ixb_stall',
        'lx_stall': 'lx_stall',
        'decode_stall': 'decode_stall',
        'dispatch_stall': 'dispatch_stall',
        'sx_stall': 'sx_stall',
        
        # Run 16
        'mem_access': 'mem_access',
        'mem_access_rd': 'mem_access_rd',
        'mem_access_wr': 'mem_access_wr'
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
        
    merged_df = pd.DataFrame(merged_rows)
    merged_df = merged_df.reset_index(drop=True)
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
        df = df.fillna(0)
        
        if 'ts' in df.columns:
            df.drop(columns=['ts'], inplace=True)
            
        if arch == "x86":
            df = merge_split_blocks(df)
            
        df = repair_dropped_samples(df)
        
        if 'sample_index' not in df.columns:
            df['sample_index'] = range(len(df))
            
    return df

def process_single_file(f, out_dir, arch="arm"):
    fname = os.path.basename(f)
    meta = parse_filename(fname)
    
    if not meta:
        return False, f"Skipped {fname} (doesn't match naming convention)", []
        
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

def align_csvs(out_dir):
    print("\n--- Starting Alignment Phase ---")
    csv_files = glob.glob(os.path.join(out_dir, "*.csv"))
    csv_files = [f for f in csv_files if not os.path.basename(f).startswith("aligned_")]
    
    if not csv_files:
        print("No CSVs found to align.")
        return

    # UPDATED: Group by CPU ID as well so ML scripts can parse it later
    groups = {}
    for f in csv_files:
        fname = os.path.basename(f)
        match = re.search(r"(?P<bench>.+)_(?P<freq>[\d\.]+)GHz_cpu(?P<cpu>\d+)_run(?P<run>\d+)_phase(?P<phase>\d+)\.csv", fname)
        if match:
            bench = match.group('bench')
            freq = match.group('freq')
            cpu = match.group('cpu')
            phase = match.group('phase')
            run = int(match.group('run')) 
            
            key = (bench, freq, cpu, phase)
            if key not in groups:
                groups[key] = []
            groups[key].append((run, f))

    for (bench, freq, cpu, phase), files_info in groups.items():
        files_info.sort(key=lambda x: x[0])
        
        aligned_df = None
        seen_columns = set()
        
        for run, f in files_info:
            df = pd.read_csv(f)
            
            if aligned_df is not None:
                cols_to_keep = ['sample_index'] + [c for c in df.columns if c not in seen_columns and c != 'sample_index']
                df = df[cols_to_keep]
            
            seen_columns.update(df.columns)
            
            if aligned_df is None:
                aligned_df = df
            else:
                aligned_df = pd.merge(aligned_df, df, on='sample_index', how='outer')
        
        cols = ['sample_index'] + sorted([c for c in aligned_df.columns if c != 'sample_index'])
        aligned_df = aligned_df[cols]

        aligned_df = aligned_df.fillna(0).astype('int64')
        aligned_df = aligned_df[aligned_df["instructions"] > 0]

        # UPDATED: Retains the CPU marker in the filename
        aligned_out = os.path.join(out_dir, f"aligned_{bench}_{freq}GHz_cpu{cpu}_phase{phase}.csv")
        aligned_df.to_csv(aligned_out, index=False)
        print(f"Created perfectly aligned trace: {aligned_out}")

def main():
    parser = argparse.ArgumentParser(description="Process raw perf .out files into CSVs in parallel and align traces.")
    
    # Defaults updated to point to the desktop folders
    parser.add_argument("--raw_dir", default="../../../raw_data/arm_desktop", help="Directory with raw .out files")
    parser.add_argument("--out_dir", default="../../../processed_data/arm_desktop", help="Directory to save CSVs")
    parser.add_argument("--jobs", type=int, default=os.cpu_count(), help="Number of parallel workers")
    parser.add_argument("--arch", choices=["x86", "arm"], default="arm", help="Target architecture (default: arm)")
    parser.add_argument("--freqs", default=None, help="Comma-separated frequencies to process (e.g. 1.0,2.0)")
    args = parser.parse_args()

    files = glob.glob(os.path.join(args.raw_dir, "**/*.out"), recursive=True)
    if args.freqs:
        freq_set = set(args.freqs.split(','))
        files = [f for f in files if any(f"_{freq}GHz_" in os.path.basename(f) for freq in freq_set)]

    file_tasks = []
    suite_out_dirs = set()
    for f in files:
        rel_dir = os.path.relpath(os.path.dirname(f), args.raw_dir)
        suite_out_dir = args.out_dir if rel_dir == '.' else os.path.join(args.out_dir, rel_dir)
        os.makedirs(suite_out_dir, exist_ok=True)
        file_tasks.append((f, suite_out_dir))
        suite_out_dirs.add(suite_out_dir)

    print(f"Found {len(files)} raw files in {args.raw_dir}")
    print(f"Processing to {args.out_dir} using {args.jobs} workers (Arch: {args.arch})...")

    count = 0
    skipped = 0

    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = {executor.submit(process_single_file, f, out, args.arch): f for f, out in file_tasks}

        for future in as_completed(futures):
            success, msg, outlier_messages = future.result()

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

    for suite_dir in sorted(suite_out_dirs):
        align_csvs(suite_dir)

if __name__ == "__main__":
    main()
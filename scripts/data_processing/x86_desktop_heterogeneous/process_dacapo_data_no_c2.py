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
    name = raw_event.rstrip(':').split(':')[0]
    if '/' in name:
        parts = name.split('/')
        name = parts[1] if len(parts) >= 2 and parts[1] else parts[0]

    name = name.lower()

    mapping = {
        'instructions': 'instructions',
        'cpu-cycles': 'cpu_cycles',
        'bus-cycles': 'bus_cycles',
        'fp_arith_inst_retired.scalar_single': 'fp_arith_scalar_single',
        'branch-loads': 'branch_loads',
        'branch-load-misses': 'branch_load_misses',
        'br_inst_retired.all_branches': 'branches',
        'br_misp_retired.all_branches': 'branch_misses',
        'ref-cycles': 'ref_cycles',
        'l1-dcache-loads': 'l1_dcache_loads',
        'l1-dcache-load-misses': 'l1_dcache_load_misses',
        'l1-dcache-stores': 'l1_dcache_stores',
        'l1-icache-load-misses': 'l1_icache_load_misses',
        'llc-loads': 'llc_loads',
        'llc-load-misses': 'llc_misses',
        'cache-references': 'cache_references',
        'cache-misses': 'cache_misses',
        'mem-loads': 'mem_loads',
        'dtlb-loads': 'dtlb_loads',
        'dtlb-load-misses': 'dtlb_load_misses',
        'itlb-load-misses': 'itlb_load_misses',
        'dtlb-stores': 'dtlb_stores',
        'dtlb-store-misses': 'dtlb_store_misses',
        'mem-stores': 'mem_stores'
    }

    if name in mapping:
        return mapping[name]
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

def parse_perf_script_output(proc_stdout, arch="x86"):
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
        return False, f"Skipped {fname} (doesn't match Dacapo naming convention)"
        
    try:
        cmd = ["perf", "script", "-i", f]
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
            df = parse_perf_script_output(proc.stdout, arch=arch)
            
        if df.empty:
            return False, f"[WARN] {fname}: No data extracted."

        out_name = f"{meta['bench']}_{meta['freq']}GHz_cpu{meta['cpu_id']}_run{meta['run']}_phase{meta['phase']}.csv"
        out_path = os.path.join(out_dir, out_name)
        
        df.to_csv(out_path, index=False)
        return True, f"Processed {fname}"
        
    except Exception as e:
        return False, f"[ERR] Failed {fname}: {e}"

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
                aligned_df = pd.merge(aligned_df, df, on='sample_index', how='inner')

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
    parser = argparse.ArgumentParser(description="Process raw Dacapo perf .out files into CSVs, stripping C2 JIT compiler samples.")
    parser.add_argument("--raw_dir", default="../../../raw_data/x86_desktop_heterogeneous", help="Directory with raw .out files")
    parser.add_argument("--out_dir", default="../../../processed_data/x86_desktop_heterogeneous_no_c2", help="Directory to save CSVs")
    parser.add_argument("--jobs", type=int, default=40, help="Number of parallel workers")
    parser.add_argument("--arch", choices=["x86", "arm"], default="x86", help="Target architecture (default: x86)")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    files = glob.glob(os.path.join(args.raw_dir, "*dacapo*.out"))
    print(f"Found {len(files)} Dacapo raw files in {args.raw_dir}")
    print(f"Processing to {args.out_dir} using {args.jobs} workers...")
    
    # Bundle tasks for the memory-safe multiprocessing map
    tasks = [(f, args.out_dir, args.arch) for f in files]
    count = 0
    
    # MAGIC FIX: maxtasksperchild=1 strictly forces process garbage collection
    with mp.Pool(processes=args.jobs, maxtasksperchild=1) as pool:
        for success, msg in pool.imap_unordered(process_single_file_wrapper, tasks):
            if success:
                count += 1
            if count > 0 and count % 50 == 0:
                print(f"  Processed {count} files...")
            
    print(f"\nDone. Processed {count} files.")
    
    align_csvs_and_evaluate(args.out_dir)

if __name__ == "__main__":
    main()
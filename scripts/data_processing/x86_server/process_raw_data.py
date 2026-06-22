import os
import glob
import pandas as pd
import re
import subprocess
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

def parse_filename(filename):
    # Handles SPEC 2017 (500-series), SPEC 2026 (700/800-series), and DaCapo:
    # all share cpu_{id}_{freq}GHz_{bench}_10000000_{run}_{phase}.out
    match = re.search(r"cpu_(?P<cpu_id>\d+)_(?P<freq>[\d\.]+)GHz_(?P<bench>.+)_10000000_(?P<run>\d+)_(?P<phase>\d+)\.out", filename)
    if match:
        return match.groupdict()
    return None

def clean_event_name(raw_event):
    name = raw_event.rstrip(':')
    name = name.split(':')[0]

    # Strip any PMU prefix (e.g. cpu_core/, cpu_atom/, uncore_imc_0/)
    if '/' in name:
        parts = name.split('/')
        if len(parts) >= 2 and parts[1]:
            name = parts[1]
        else:
            name = parts[0]

    name = name.lower()

    # Mapping for x86 server PMU counters (runs 0-12 of x86_server.cfg)
    mapping = {
        'instructions':                              'instructions',
        'cpu-cycles':                                'cpu_cycles',
        'bus-cycles':                                'bus_cycles',
        'ref-cycles':                                'ref_cycles',

        'fp_arith_inst_retired.scalar_single':       'fp_arith_scalar_single',
        'fp_arith_inst_retired.scalar_double':       'fp_arith_scalar_double',
        'fp_arith_inst_retired.128b_packed_single':  'fp_arith_128b_packed_single',
        'fp_arith_inst_retired.128b_packed_double':  'fp_arith_128b_packed_double',
        'fp_arith_inst_retired.256b_packed_single':  'fp_arith_256b_packed_single',

        'branch-loads':                              'branch_loads',
        'branch-load-misses':                        'branch_load_misses',
        'branch-instructions':                       'branch_instructions',
        'branch-misses':                             'branch_misses',
        'br_inst_retired.all_branches':              'branches',
        'br_misp_retired.all_branches':              'branch_misses',

        'l1-dcache-loads':                           'l1_dcache_loads',
        'l1-dcache-load-misses':                     'l1_dcache_load_misses',
        'l1-dcache-stores':                          'l1_dcache_stores',
        'l1-icache-load-misses':                     'l1_icache_load_misses',

        'llc-loads':                                 'llc_loads',
        'llc-load-misses':                           'llc_misses',

        'cache-references':                          'cache_references',
        'cache-misses':                              'cache_misses',
        'mem-loads':                                 'mem_loads',
        'mem-stores':                                'mem_stores',

        'dtlb-loads':                                'dtlb_loads',
        'dtlb-load-misses':                          'dtlb_load_misses',
        'dtlb-stores':                               'dtlb_stores',
        'dtlb-store-misses':                         'dtlb_store_misses',
        'itlb-load-misses':                          'itlb_load_misses',

        'node-loads':                                'node_loads',
        'node-load-misses':                          'node_load_misses',
        'slots':                                     'slots',

        'page-faults':                               'page_faults',
        'minor-faults':                              'minor_faults',
        'major-faults':                              'major_faults',
        'context-switches':                          'context_switches',
    }

    if name in mapping:
        return mapping[name]

    # Fallback for any other events (replaces dashes/dots with underscores)
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
    std_val  = instrs_to_check.std()
    min_val  = instrs_to_check.min()
    max_val  = instrs_to_check.max()

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

        ts_str    = parts[ts_idx].replace(':', '')
        val_str   = parts[ts_idx + 1].replace(',', '')
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

def process_single_file(f, out_dir, arch="x86"):
    fname = os.path.basename(f)
    meta  = parse_filename(fname)

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

    # Single-CPU setup: group by (bench, freq, phase) only
    groups = {}
    for f in csv_files:
        fname = os.path.basename(f)
        match = re.search(r"(?P<bench>.+)_(?P<freq>[\d\.]+)GHz_cpu(?P<cpu>\d+)_run(?P<run>\d+)_phase(?P<phase>\d+)\.csv", fname)
        if match:
            key = (match.group('bench'), match.group('freq'), match.group('phase'))
            groups.setdefault(key, []).append((int(match.group('run')), f))

    for (bench, freq, phase), files_info in groups.items():
        files_info.sort(key=lambda x: x[0])

        aligned_df  = None
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
                aligned_df = pd.merge(aligned_df, df, on='sample_index', how='left')

        cols = ['sample_index'] + sorted([c for c in aligned_df.columns if c != 'sample_index'])
        aligned_df = aligned_df[cols]

        aligned_df = aligned_df.fillna(0).astype('int64')
        aligned_df = aligned_df[aligned_df["instructions"] > 0]

        aligned_out = os.path.join(out_dir, f"aligned_{bench}_{freq}GHz_phase{phase}.csv")
        aligned_df.to_csv(aligned_out, index=False)
        print(f"Created perfectly aligned trace: {aligned_out}")

def main():
    parser = argparse.ArgumentParser(description="Process raw perf .out files into CSVs in parallel and align traces.")

    parser.add_argument("--raw_dir", default="../../../raw_data/x86_server", help="Directory with raw .out files")
    parser.add_argument("--out_dir", default="../../../processed_data_10M/x86_server", help="Directory to save CSVs")
    parser.add_argument("--jobs",    type=int, default=os.cpu_count(), help="Number of parallel workers")
    parser.add_argument("--arch",    choices=["x86", "arm"], default="x86", help="Target architecture (default: x86)")
    parser.add_argument("--freqs",  default=None, help="Comma-separated frequencies to process (e.g. 1.0,2.0)")
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

    count   = 0
    skipped = 0

    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = {executor.submit(process_single_file, f, out, args.arch): f for f, out in file_tasks}

        for future in as_completed(futures):
            success, msg, outlier_messages = future.result()

            if success:
                count += 1
            else:
                skipped += 1

            for out_msg in outlier_messages:
                print(out_msg)

            if count > 0 and count % 20 == 0:
                print(f"  Processed {count} files...")

    print(f"\nDone. Processed {count} files. Skipped {skipped}.")

    for suite_dir in sorted(suite_out_dirs):
        align_csvs(suite_dir)

if __name__ == "__main__":
    main()

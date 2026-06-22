#!/usr/bin/env python3
"""
build_power_aligned_traces.py -- build per-block power columns for the
SPEC CPU2017 my_run=100 power-trace collection.

For each cpu_<cpu>_<freq>GHz_<bench>_10000000_100_<phase>.out in --power_dir,
aligns the perf-record instruction/cycle blocks with the RAPL power trace
(using the helpers from analyze_block_granularity.py), computes idle-relative
and idle-offset ("total") power columns, and writes the result.

Two modes:
  - standalone (--aligned_dir not given): write an 8-column
    aligned_<bench>_<freq>GHz_cpu<cpu>_phase<phase>.csv with
    sample_index/instructions/cpu_cycles/ref_cycles + the 4 power columns,
    in the format expected by
    scripts/evaluation/scheduling/utils/generate_speedup_matrix.py.
  - merge (--aligned_dir given): look for an existing
    aligned_<bench>_<freq>GHz_cpu<cpu>_phase<phase>.csv in --aligned_dir
    (a full hardware-counter aligned trace) and merge the 4 power columns
    into it on sample_index. Files with no matching trace are skipped.

Each .out file is processed independently, so --workers > 1 runs them in
a process pool.
"""
import argparse
import glob
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    '../../data_collection/x86_desktop_heterogeneous/power_collection/instr_block_power')))
from analyze_block_granularity import (
    parse_perf_script_output, parse_power_data, read_sync_time,
    get_idle_power, align_perf_power, add_rolling_power,
)
from process_raw_data import merge_split_blocks, repair_dropped_samples

FILENAME_RE = re.compile(
    r"cpu_(?P<cpu>\d+)_(?P<freq>[\d.]+)GHz_(?P<bench>.+)_10000000_100_(?P<phase>\d+)\.out")

ROLLING_N = 10
MATCH_RATE_WARN_THRESHOLD = 0.90


def check_block_correction(df_perf):
    """Return True if merge_split_blocks/repair_dropped_samples (the
    process_raw_data.py correction logic) would change the block count."""
    counters = df_perf.drop(columns=['perf_time_s', 'block_index'], errors='ignore').copy()
    counters['sample_index'] = range(len(counters))
    corrected = merge_split_blocks(counters)
    corrected = repair_dropped_samples(corrected)
    return len(corrected) != len(df_perf)


def process_file(out_file, power_dir, aligned_dir, output_dir):
    """Process a single .out file. Returns (status, messages, out_name)
    where status is one of 'written', 'skipped', and messages is a list
    of log lines to print."""
    msgs = []
    fname = os.path.basename(out_file)
    match = FILENAME_RE.match(fname)
    if not match:
        msgs.append(f"  [SKIP] {fname}: doesn't match naming convention")
        return 'skipped', msgs, None

    cpu = match.group('cpu')
    freq = match.group('freq')
    bench = match.group('bench')
    phase = match.group('phase')

    base = out_file[:-len('.out')]
    power_csv = base + '_power.csv'
    sync_file = base + '_sync.txt'
    idle_name = f'idle_{cpu}_{freq}GHz_power.csv'
    idle_csv = os.path.join(os.path.dirname(out_file), idle_name)
    if not os.path.exists(idle_csv):
        idle_csv = os.path.join(power_dir, idle_name)

    if not (os.path.exists(power_csv) and os.path.exists(sync_file) and os.path.exists(idle_csv)):
        msgs.append(f"  [SKIP] {fname}: missing power/sync/idle file")
        return 'skipped', msgs, None

    idle_w = get_idle_power(idle_csv)
    sync_time = read_sync_time(sync_file)
    df_perf = parse_perf_script_output(out_file)
    df_power = parse_power_data(power_csv)

    if df_perf.empty or 'ref_cycles' not in df_perf.columns:
        msgs.append(f"  [SKIP] {fname}: no ref_cycles column")
        return 'skipped', msgs, None

    warn_correction = check_block_correction(df_perf)
    if warn_correction:
        msgs.append(f"  [WARN] {fname}: block count would change under "
                     f"merge_split_blocks/repair_dropped_samples")

    df_aligned = align_perf_power(df_perf, df_power, sync_time)
    if df_aligned.empty:
        msgs.append(f"  [SKIP] {fname}: alignment produced no overlap")
        return 'skipped', msgs, None

    match_rate = len(df_aligned) / len(df_perf)
    warn_match = match_rate < MATCH_RATE_WARN_THRESHOLD
    if warn_match:
        msgs.append(f"  [WARN] {fname}: only {100*match_rate:.1f}% of blocks "
                     f"matched a power sample")

    df_aligned = add_rolling_power(df_aligned, ROLLING_N, idle_w)
    df_aligned = df_aligned.rename(columns={'block_index': 'sample_index'})
    df_aligned['power_watts_total_block'] = df_aligned['power_watts_block'] + idle_w
    df_aligned[f'power_watts_total_rolling_{ROLLING_N}'] = (
        df_aligned[f'power_watts_rolling_{ROLLING_N}'] + idle_w)

    out_name = f"aligned_{bench}_{freq}GHz_cpu{cpu}_phase{phase}.csv"

    if aligned_dir is None:
        cols = ['sample_index', 'instructions', 'cpu_cycles', 'ref_cycles',
                'power_watts_block', f'power_watts_rolling_{ROLLING_N}',
                'power_watts_total_block', f'power_watts_total_rolling_{ROLLING_N}']
        out_df = df_aligned[cols]
    else:
        matches = glob.glob(os.path.join(aligned_dir, '**', out_name), recursive=True)
        if not matches:
            msgs.append(f"  [SKIP] {fname}: no matching aligned trace {out_name} under {aligned_dir}")
            return 'skipped', msgs, None
        aligned_path = matches[0]

        power_cols = df_aligned[
            ['sample_index', 'power_watts_block', f'power_watts_rolling_{ROLLING_N}',
             'power_watts_total_block', f'power_watts_total_rolling_{ROLLING_N}']]

        df_trace = pd.read_csv(aligned_path)
        out_df = pd.merge(df_trace, power_cols, on='sample_index', how='left')

        n_nan = out_df['power_watts_block'].isna().sum()
        if n_nan:
            # RAPL power sampling stops a few seconds before perf record
            # finishes, leaving a contiguous trailing run of unmatched
            # blocks. Forward-fill from the last matched (steady-state)
            # power sample rather than leaving these as NaN.
            power_col_names = ['power_watts_block', f'power_watts_rolling_{ROLLING_N}',
                                'power_watts_total_block', f'power_watts_total_rolling_{ROLLING_N}']
            out_df[power_col_names] = out_df[power_col_names].ffill()
            msgs.append(f"  [INFO] {out_name}: {n_nan}/{len(out_df)} rows had no power data "
                         f"(tail blocks) -- forward-filled from last sample")

    out_path = os.path.join(output_dir, out_name)
    out_df.to_csv(out_path, index=False)

    status = 'written'
    if warn_match:
        status += ',warn_match'
    if warn_correction:
        status += ',warn_correction'
    return status, msgs, out_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--power_dir', default='power_data_10M/x86_desktop_heterogeneous')
    parser.add_argument('--aligned_dir', default=None,
                         help="if given, merge power columns into existing aligned_*.csv "
                              "traces here instead of writing standalone files")
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--workers', type=int, default=1,
                         help="number of .out files to process in parallel (default 1)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    out_files = sorted(glob.glob(os.path.join(args.power_dir, '**', 'cpu_*_*GHz_*_10000000_100_*.out'), recursive=True))
    print(f"Found {len(out_files)} my_run=100 .out files in {args.power_dir}")

    n_written = 0
    n_skipped = 0
    n_warn_match = 0
    n_warn_correction = 0

    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(process_file, out_file, args.power_dir,
                                 args.aligned_dir, args.output_dir): out_file
                for out_file in out_files
            }
            for future in as_completed(futures):
                status, msgs, out_name = future.result()
                for m in msgs:
                    print(m)
                if status.startswith('written'):
                    n_written += 1
                    if 'warn_match' in status:
                        n_warn_match += 1
                    if 'warn_correction' in status:
                        n_warn_correction += 1
                else:
                    n_skipped += 1
    else:
        for out_file in out_files:
            status, msgs, out_name = process_file(out_file, args.power_dir,
                                                    args.aligned_dir, args.output_dir)
            for m in msgs:
                print(m)
            if status.startswith('written'):
                n_written += 1
                if 'warn_match' in status:
                    n_warn_match += 1
                if 'warn_correction' in status:
                    n_warn_correction += 1
            else:
                n_skipped += 1

    print(f"\nWrote {n_written} aligned_*.csv files to {args.output_dir}")
    print(f"  {n_skipped} files skipped")
    print(f"  {n_warn_match} files with <{int(100*MATCH_RATE_WARN_THRESHOLD)}% power match rate")
    print(f"  {n_warn_correction} files where block correction would change row count")


if __name__ == '__main__':
    main()

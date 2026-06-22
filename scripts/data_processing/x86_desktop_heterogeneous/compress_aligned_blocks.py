#!/usr/bin/env python3
"""
compress_aligned_blocks.py -- merge N consecutive 10M-instruction blocks of
an aligned_*.csv (with power_watts_total_block/power_watts_total_rolling_10)
into coarser N*10M-instruction blocks.

For each input file:
  - drop trailing rows with NaN power (the unmatched tail blocks)
  - truncate to a multiple of N (drop an incomplete final group)
  - group every N consecutive rows (sample_index // N):
      - sum all additive hardware-counter columns (instructions, cpu_cycles,
        ref_cycles, and any others present)
      - power_watts_total_block = ref_cycles-weighted average of
        power_watts_total_block over the group (a true time-weighted
        average, since ref_cycles ticks at a fixed clock and is a
        proportional measure of wall-clock duration)
      - power_watts_block = power_watts_total_block - idle_w (recomputed,
        not averaged)
      - power_watts_rolling_10 / power_watts_total_rolling_10 recomputed via
        a fresh centered rolling-10 average over the new (coarser) series
  - new sample_index = 0..M-1

idle_w is looked up per (cpu, freq) from
power_data_10M/x86_desktop_heterogeneous/idle_<cpu>_<freq>GHz_power.csv.
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    '../../data_collection/x86_desktop_heterogeneous/power_collection/instr_block_power')))
from analyze_block_granularity import get_idle_power

PERFCOUNT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../..'))
IDLE_DIR = os.path.join(PERFCOUNT_ROOT, 'power_data_10M/x86_desktop_heterogeneous')

ALIGNED_RE = re.compile(r"aligned_.+_(?P<freq>[\d.]+)GHz_cpu(?P<cpu>\d+)_phase\d+\.csv")
ROLLING_N = 10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir')
    parser.add_argument('output_dir')
    parser.add_argument('--factor', type=int, default=10,
                         help="number of 10M-instruction blocks to merge (default 10 -> 100M)")
    args = parser.parse_args()
    N = args.factor

    os.makedirs(args.output_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.input_dir, '**', 'aligned_*.csv'), recursive=True))
    print(f"Found {len(files)} aligned_*.csv files in {args.input_dir}")

    idle_cache = {}
    n_written = 0
    n_skipped = 0

    for path in files:
        fname = os.path.basename(path)
        match = ALIGNED_RE.match(fname)
        if not match:
            print(f"  [SKIP] {fname}: doesn't match naming convention")
            n_skipped += 1
            continue

        cpu = match.group('cpu')
        freq = match.group('freq')

        key = (cpu, freq)
        if key not in idle_cache:
            idle_name = f"idle_{cpu}_{freq}GHz_power.csv"
            idle_csv = os.path.join(os.path.dirname(path), idle_name)
            if not os.path.exists(idle_csv):
                idle_csv = os.path.join(IDLE_DIR, idle_name)
            if not os.path.exists(idle_csv):
                print(f"  [SKIP] {fname}: no idle baseline {idle_name}")
                n_skipped += 1
                continue
            idle_cache[key] = get_idle_power(idle_csv)
        idle_w = idle_cache[key]

        df = pd.read_csv(path)
        required = {'sample_index', 'ref_cycles', 'power_watts_total_block'}
        if not required.issubset(df.columns):
            print(f"  [SKIP] {fname}: missing required columns {required - set(df.columns)}")
            n_skipped += 1
            continue

        df = df.dropna(subset=['power_watts_total_block']).reset_index(drop=True)

        n_groups = len(df) // N
        if n_groups == 0:
            print(f"  [SKIP] {fname}: fewer than {N} valid blocks ({len(df)})")
            n_skipped += 1
            continue
        df = df.iloc[:n_groups * N].copy()

        group_id = np.arange(len(df)) // N

        sum_cols = [c for c in df.columns
                     if c != 'sample_index' and not c.startswith('power_watts')]

        out = df.groupby(group_id)[sum_cols].sum()

        # ref_cycles-weighted average power (time-weighted)
        weighted_power = (df['power_watts_total_block'] * df['ref_cycles']).groupby(group_id).sum() \
            / df['ref_cycles'].groupby(group_id).sum()
        out['power_watts_total_block'] = weighted_power
        out['power_watts_block'] = out['power_watts_total_block'] - idle_w

        out = out.reset_index(drop=True)
        out['sample_index'] = range(len(out))

        out['power_watts_rolling_10'] = out['power_watts_block'].rolling(
            window=ROLLING_N, center=True, min_periods=1).mean()
        out['power_watts_total_rolling_10'] = out['power_watts_total_block'].rolling(
            window=ROLLING_N, center=True, min_periods=1).mean()

        cols = ['sample_index'] + [c for c in sum_cols if c != 'sample_index'] + \
               ['power_watts_block', 'power_watts_rolling_10',
                'power_watts_total_block', 'power_watts_total_rolling_10']
        out = out[cols]

        rel_path = os.path.relpath(path, args.input_dir)
        out_path = os.path.join(args.output_dir, rel_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        out.to_csv(out_path, index=False)
        n_written += 1

    print(f"\nWrote {n_written} compressed aligned_*.csv files (factor={N}) to {args.output_dir}")
    print(f"  {n_skipped} files skipped")


if __name__ == '__main__':
    main()

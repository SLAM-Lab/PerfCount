#!/usr/bin/env python3
"""analyze_block_granularity.py  (RECONSTRUCTED)

Helpers for aligning perf-record instruction blocks (my_run=100 power trace)
with the co-recorded RAPL energy-cores samples, used by
build_power_aligned_traces.py.

This file was lost; it is reconstructed from its call sites in
build_power_aligned_traces.py and the perf-script / perf-stat data formats.
Two conventions could not be recovered from the call sites and are chosen to
be physically sensible + validated empirically (match-rate + power magnitudes):

  * Clock alignment: perf-record sample timestamps (from `perf script`) and the
    RAPL perf-stat sample times share CLOCK_MONOTONIC; `sync_time` (from the
    _sync.txt) is the monotonic time at which the RAPL `perf stat -I` began, so
    RAPL sample i covers monotonic time `sync_time + time_s[i]`. Each perf block
    is assigned the power of the RAPL interval containing its timestamp.
  * `power_watts_block` is the raw measured average core power during the block
    (absolute). build_power_aligned_traces adds `idle_w` to form the "total"
    columns; `power_watts_block` here is therefore the measured value it offsets.

Functions:
  parse_perf_script_output(out_file)  -> df[block_index, instructions, cpu_cycles,
                                            ref_cycles, perf_time_s]
  parse_power_data(power_csv)         -> df[time_s, power_w]
  read_sync_time(sync_file)           -> float (monotonic seconds)
  get_idle_power(idle_csv)            -> float (watts, median idle core power)
  align_perf_power(df_perf, df_power, sync_time) -> df_perf + power_watts_block
  add_rolling_power(df, n, idle_w=0)  -> df + power_watts_rolling_<n>
"""
import subprocess
import numpy as np
import pandas as pd

_EVENT_MAP = {
    'instructions': 'instructions',
    'cpu-cycles': 'cpu_cycles', 'cycles': 'cpu_cycles',
    'ref-cycles': 'ref_cycles',
}


def _clean_event(raw):
    name = raw.rstrip(':').split(':')[0]
    if '/' in name:                       # cpu_atom/instructions/ -> instructions
        parts = [p for p in name.split('/') if p]
        name = parts[1] if len(parts) >= 2 else parts[0]
    return _EVENT_MAP.get(name.lower(), name.lower())


def parse_perf_script_output(out_file, arch="x86"):
    """Run `perf script` on the binary .out and group samples into per-block rows,
    keeping the block's monotonic timestamp as perf_time_s. No block correction."""
    proc = subprocess.run(["perf", "script", "-i", str(out_file)],
                          stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    data, cur = [], {}
    for raw in proc.stdout.decode('utf-8', errors='replace').splitlines():
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        ts_idx = -1
        for i, p in enumerate(parts):
            if p.endswith(':'):
                try:
                    float(p[:-1]); ts_idx = i; break
                except ValueError:
                    continue
        if ts_idx == -1 or ts_idx + 2 >= len(parts):
            continue
        ts = parts[ts_idx][:-1]
        val_str = parts[ts_idx + 1].replace(',', '')
        event = _clean_event(parts[ts_idx + 2])
        try:
            value = int(val_str)
        except ValueError:
            if val_str == '<not':
                value = 0
            else:
                continue
        if 'ts' in cur and cur['ts'] != ts:
            data.append(cur); cur = {}
        cur['ts'] = ts
        cur[event] = value
    if cur:
        data.append(cur)
    df = pd.DataFrame(data)
    if df.empty:
        return df
    df['perf_time_s'] = df['ts'].astype(float)
    df.drop(columns=['ts'], inplace=True)
    df.insert(0, 'block_index', range(len(df)))
    for c in ('instructions', 'cpu_cycles', 'ref_cycles'):
        if c not in df.columns:
            df[c] = 0
    return df


def parse_power_data(power_csv):
    """Parse a `perf stat -I -x,` energy-cores CSV into per-interval power (W).
    Row: time_s, energy_J, unit, event, counter_run_ns, pct, ..."""
    times, energies = [], []
    with open(power_csv) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            if len(parts) < 4:
                continue
            if 'energy' not in parts[3].lower():
                continue
            try:
                times.append(float(parts[0]))
                energies.append(float(parts[1]))
            except ValueError:
                continue
    if not times:
        return pd.DataFrame(columns=['time_s', 'power_w'])
    t = np.asarray(times); e = np.asarray(energies)
    dt = np.diff(t, prepend=0.0)
    dt[0] = t[0] if t[0] > 0 else (dt[1] if len(dt) > 1 else 1.0)
    dt[dt <= 0] = np.nan
    power = e / dt
    df = pd.DataFrame({'time_s': t, 'power_w': power})
    df['power_w'] = df['power_w'].bfill().ffill()
    return df


def read_sync_time(sync_file):
    with open(sync_file) as f:
        return float(f.read().strip())


def get_idle_power(idle_csv):
    dfp = parse_power_data(idle_csv)
    if dfp.empty:
        return 0.0
    return float(np.nanmedian(dfp['power_w'].values))


def align_perf_power(df_perf, df_power, sync_time, idle_w=0.0):
    """Assign each perf block the RAPL core power of the interval containing its
    monotonic timestamp (sync_time + RAPL elapsed). Blocks with no covering RAPL
    sample (e.g. before RAPL started / after it stopped) are dropped.

    Returns power_watts_block = idle-RELATIVE (active) power = measured - idle_w,
    so that the caller's `power_watts_total_block = block + idle_w` recovers the
    absolute measured power. Pass idle_w=0 to get raw measured power."""
    if df_perf.empty or df_power.empty:
        return pd.DataFrame()
    rapl_mono_end = sync_time + df_power['time_s'].values          # interval END, monotonic
    rapl_pow = df_power['power_w'].values
    bt = df_perf['perf_time_s'].values
    idx = np.searchsorted(rapl_mono_end, bt, side='left')          # first interval ending >= block ts
    out = df_perf.copy()
    p = np.full(len(bt), np.nan)
    good = idx < len(rapl_pow)
    p[good] = rapl_pow[idx[good]] - idle_w                         # idle-relative (active)
    out['power_watts_block'] = p
    out = out[np.isfinite(out['power_watts_block'].values)].reset_index(drop=True)
    return out


def add_rolling_power(df_aligned, n, idle_w=0.0):
    df = df_aligned.copy()
    df[f'power_watts_rolling_{n}'] = (
        df['power_watts_block'].rolling(n, min_periods=1).mean())
    return df

# temporal_policies.py
#
# Supplies the "vector of counters" (a window of cost/proxy data) that a
# decision-making policy (see decision_policies.py) gets to see at chunk `i`.
# This is the TEMPORAL axis of the policy grid: what data is visible, not how
# it is used.
import numpy as np


def reactive_window_raw(time_mat, energy_mat, i, window_size):
    """Past `window_size` chunks (exclusive of chunk i). Empty at i==0."""
    start = max(0, i - window_size)
    return time_mat[start:i, :], energy_mat[start:i, :]


def oracle_window_raw(time_mat, energy_mat, i, window_size):
    """Current + future chunks, real measured data ("free" lookahead).

    Always returns at least one row (the current chunk).
    """
    n_chunks = len(time_mat)
    window_len = min(window_size, n_chunks - i)
    return time_mat[i:i + window_len, :], energy_mat[i:i + window_len, :]


def reactive_oracle_window_raw(time_mat, energy_mat, i):
    """Prior chunk's TRUE timings used to decide chunk i.

    'Reactive oracle': perfect knowledge of what each config achieved at
    chunk i-1, but cannot see chunk i before running it.  Returns empty
    arrays at i==0 (no prior chunk exists yet).
    """
    if i == 0:
        return time_mat[0:0, :], energy_mat[0:0, :]
    return time_mat[i - 1:i, :], energy_mat[i - 1:i, :]


def reactive_signal_window(proxy_signal, i, window_size):
    """Past `window_size` proxy-signal values (exclusive of chunk i).

    This is the "performance counter" input used by SOTA/heuristic governors
    (ondemand, conservative, schedutil, EWMA, Thread Director, etc.) - they
    only ever see past proxy-signal samples, never the cost-matrix windows
    used by decide_greedy/decide_mpc. Empty at i==0.
    """
    start = max(0, i - window_size)
    return proxy_signal[start:i]

# Note: workload forecasting is NOT implemented here. It is applied upstream, by
# swapping in a forecast tensor whose row i holds a causal forecast of chunk i
# (see main.py's *_forecast_dir handling). The decision functions are then reused
# unchanged, since they only care about the window's shape, not whether it holds
# measured or predicted values. A "Forecast x Global" cell does not exist by
# design: a full-trace Viterbi DP requires ground truth for every chunk, which is
# exactly what an imperfect prediction cannot supply.

#!/usr/bin/env python3
"""
benchmark_backends.py
=====================
Formal inference latency benchmark across three execution backends:
  1. CatBoost Python   — model.predict()  (current production baseline)
  2. ONNX Runtime      — onnxruntime session (thin Python layer)
  3. C++ standalone    — CatBoost-generated .cpp compiled at -O3 + -march=native

For each (exp_type, platform, gran, suite, feature_set):
  • Picks one representative trained model (first alphabetically)
  • Benchmarks all three backends with single-row inference, single thread
  • Reads mean WMAPE from grand_summary.csv
  • Writes one row to results/cross_platform/inference/backends_{gran}.csv

C++ .so files are cached in --cache_dir to avoid recompilation on re-runs.

Usage
-----
  python benchmark_backends.py                   # 10M granularity
  python benchmark_backends.py --gran 100M
  python benchmark_backends.py --reps 200000 --cpu 2
"""

import argparse
import ctypes
import gc
import glob
import hashlib
import logging
import os
import subprocess
import tempfile
import textwrap
import time

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
import onnxruntime as ort

# Suppress ONNX Runtime shape-mismatch warnings (benign for scalar regression)
ort.set_default_logger_severity(3)   # ERROR only

FEATURE_SETS   = ["full", "top6", "top4"]
EXP_TYPES      = ["cross_freq", "cross_proc", "cross_sys"]
REPS_TARGET_US = 500_000
MIN_REPS       = 1_000_000
MAX_REPS       = 1_000_000
WARMUP         = 1_000


# ---------------------------------------------------------------------------
# Model discovery (same logic as measure_inference_latency.py)
# ---------------------------------------------------------------------------

def discover_models(results_root, gran, feature_sets, platform_filter=None):
    """
    platform_filter, if given, is matched exactly against the platform
    directory name (e.g. "x86_10M") -- this is how callers restrict
    discovery to a single platform, such as the x86 desktop (P-Core/E-Core)
    for cross_freq/cross_proc, which also happens to be the right filter
    for cross_sys since that experiment bundles server<->desktop under the
    same "x86_{gran}" directory.
    """
    groups = {}
    for exp_type in EXP_TYPES:
        exp_dir = os.path.join(results_root, exp_type)
        if not os.path.isdir(exp_dir):
            continue
        for cbm in sorted(glob.glob(os.path.join(exp_dir, "**", "model_*.cbm"),
                                    recursive=True)):
            rel   = os.path.relpath(cbm, exp_dir)
            parts = rel.split(os.sep)
            if not parts[0].endswith(f"_{gran}"):
                continue
            platform = parts[0]
            if platform_filter is not None and platform != platform_filter:
                continue
            for i, p in enumerate(parts[1:], start=1):
                if p in feature_sets:
                    suite = parts[i - 1] if i >= 2 else "unknown"
                    key   = (exp_type, platform, suite, p)
                    groups.setdefault(key, cbm)
                    break

    return [
        {"exp_type": k[0], "platform": k[1], "suite": k[2],
         "feature_set": k[3], "model_path": v}
        for k, v in sorted(groups.items())
    ]


# ---------------------------------------------------------------------------
# Accuracy: read grand_summary.csv
# ---------------------------------------------------------------------------

def load_accuracy(model_path, results_root):
    """Return (mean wmape_ml, mean mape_ml) from grand_summary.csv.
    cross_sys grand_summary has only mape_ml; wmape_ml returns NaN there."""
    feature_dir = os.path.dirname(os.path.dirname(model_path))
    gs_path     = os.path.join(feature_dir, "grand_summary.csv")
    if not os.path.exists(gs_path):
        return float("nan"), float("nan")
    gs    = pd.read_csv(gs_path)
    wmape = gs["wmape_ml"].mean() if "wmape_ml" in gs.columns else float("nan")
    mape  = gs["mape_ml"].mean()  if "mape_ml"  in gs.columns else float("nan")
    return wmape, mape


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def adaptive_reps(call_fn, warmup):
    for _ in range(warmup // 5):
        call_fn()
    t0 = time.perf_counter_ns()
    for _ in range(1000):
        call_fn()
    mean_us = (time.perf_counter_ns() - t0) / 1e3 / 1000
    reps = int(REPS_TARGET_US / max(mean_us, 0.01))
    return max(MIN_REPS, min(MAX_REPS, reps))


def time_fn(call_fn, reps, warmup):
    for _ in range(warmup):
        call_fn()
    gc.disable()
    buf = np.empty(reps, dtype=np.int64)
    for i in range(reps):
        t0    = time.perf_counter_ns()
        call_fn()
        buf[i] = time.perf_counter_ns() - t0
    gc.enable()
    us = buf / 1e3
    return {
        "mean_us":   float(np.mean(us)),
        "median_us": float(np.median(us)),
        "p90_us":    float(np.percentile(us, 90)),
        "p99_us":    float(np.percentile(us, 99)),
        "min_us":    float(np.min(us)),
        "n_reps":    reps,
    }


# ---------------------------------------------------------------------------
# Backend 1: CatBoost Python
# ---------------------------------------------------------------------------

def bench_catboost(model, X):
    reps = adaptive_reps(lambda: model.predict(X, thread_count=1), WARMUP)
    return time_fn(lambda: model.predict(X, thread_count=1), reps, WARMUP)


# ---------------------------------------------------------------------------
# Backend 2: ONNX Runtime
# ---------------------------------------------------------------------------

def bench_onnx(model, X, tmp_dir):
    onnx_path = os.path.join(tmp_dir, "model.onnx")
    model.save_model(onnx_path, format="onnx")

    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = 1
    opts.log_severity_level   = 3   # suppress shape warnings
    sess     = ort.InferenceSession(onnx_path, sess_options=opts,
                                    providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    feed     = {inp_name: X.astype(np.float32)}

    reps = adaptive_reps(lambda: sess.run(None, feed), WARMUP)
    return time_fn(lambda: sess.run(None, feed), reps, WARMUP)


# ---------------------------------------------------------------------------
# Backend 3: C++ standalone (compiled)
# ---------------------------------------------------------------------------

def _model_hash(model_path):
    h = hashlib.md5(model_path.encode())
    h.update(str(os.path.getmtime(model_path)).encode())
    return h.hexdigest()[:12]


def build_cpp_so(model, model_path, cache_dir):
    """
    Export model to .cpp, wrap with extern "C" shim, compile to .so.
    Returns path to compiled .so, or None on failure.
    Uses a hash-keyed cache to avoid recompiling unchanged models.
    """
    key     = _model_hash(model_path)
    so_path = os.path.join(cache_dir, f"{key}.so")
    if os.path.exists(so_path):
        return so_path

    cpp_path  = os.path.join(cache_dir, f"{key}_model.cpp")
    shim_path = os.path.join(cache_dir, f"{key}_shim.cpp")

    model.save_model(cpp_path, format="cpp")

    shim = textwrap.dedent(f"""\
        #include <vector>
        #include "{cpp_path}"
        extern "C" {{
            double predict_single(const float* x, int n) {{
                return ApplyCatboostModel(std::vector<float>(x, x + n));
            }}
        }}
    """)
    with open(shim_path, "w") as f:
        f.write(shim)

    result = subprocess.run(
        ["g++", "-O3", "-march=native", "-std=c++14",
         "-shared", "-fPIC", "-o", so_path, shim_path],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        logging.warning("C++ compile failed for %s:\n%s", model_path, result.stderr)
        return None
    return so_path


def bench_cpp(model, model_path, X, cache_dir):
    so_path = build_cpp_so(model, model_path, cache_dir)
    if so_path is None:
        return None

    lib = ctypes.CDLL(so_path)
    lib.predict_single.restype  = ctypes.c_double
    lib.predict_single.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int]
    n        = X.shape[1]
    x_ptr    = X.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    call_fn  = lambda: lib.predict_single(x_ptr, n)

    reps = adaptive_reps(call_fn, WARMUP)
    return time_fn(call_fn, reps, WARMUP)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gran",         default="10M")
    ap.add_argument("--results_root", default="results/cross_platform")
    ap.add_argument("--out_dir",      default="results/cross_platform/inference")
    ap.add_argument("--cache_dir",    default="/tmp/inference_so_cache",
                    help="Directory for cached compiled .so files.")
    ap.add_argument("--cpu",          type=int, default=0,
                    help="CPU core to pin to.")
    ap.add_argument("--exp-types",    default=None,
                    help="Comma-separated experiment types to run (default: all).")
    ap.add_argument("--platform",     default="x86",
                    help="Restrict discovery to this platform prefix, matched as "
                         "'<platform>_<gran>' (default: 'x86', i.e. the x86 desktop "
                         "P-Core/E-Core pair for cross_freq/cross_proc, and the "
                         "combined x86 server<->desktop directory for cross_sys). "
                         "Pass 'all' to disable filtering and discover every platform.")
    ap.add_argument("--min-reps",     type=int, default=None,
                    help="Override MIN_REPS.")
    ap.add_argument("--max-reps",     type=int, default=None,
                    help="Override MAX_REPS.")
    ap.add_argument("--append",       action="store_true",
                    help="Append to existing output CSV instead of overwriting.")
    args = ap.parse_args()

    global MIN_REPS, MAX_REPS, EXP_TYPES
    if args.min_reps is not None:
        MIN_REPS = args.min_reps
    if args.max_reps is not None:
        MAX_REPS = args.max_reps
    if args.exp_types is not None:
        EXP_TYPES = [e.strip() for e in args.exp_types.split(",")]

    try:
        os.sched_setaffinity(0, {args.cpu})
        print(f"Pinned to CPU {args.cpu}")
    except (AttributeError, OSError) as e:
        print(f"[WARN] CPU pin failed: {e}")

    os.makedirs(args.out_dir,   exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    platform_filter = None if args.platform == "all" else f"{args.platform}_{args.gran}"
    entries = discover_models(args.results_root, args.gran, FEATURE_SETS, platform_filter)
    if not entries:
        print("No models found. Check --results_root, --gran, and --platform.")
        return
    print(f"\nFound {len(entries)} groups at {args.gran}"
          f" (platform filter: {platform_filter or 'none'})\n")

    rows = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        for entry in entries:
            label = (f"{entry['exp_type']}/{entry['platform']}/"
                     f"{entry['suite']}/{entry['feature_set']}")
            print(f"[{label}]")

            model = CatBoostRegressor()
            model.load_model(entry["model_path"])
            n_feat  = len(model.feature_names_)
            n_trees = model.tree_count_
            X       = np.ones((1, n_feat), dtype=np.float32)

            wmape, mape = load_accuracy(entry["model_path"], args.results_root)

            base = {
                "exp_type":    entry["exp_type"],
                "platform":    entry["platform"],
                "suite":       entry["suite"],
                "feature_set": entry["feature_set"],
                "n_features":  n_feat,
                "n_trees":     n_trees,
                "wmape_ml":    wmape,
                "mape_ml":     mape,
            }

            # --- CatBoost Python ---
            stats = bench_catboost(model, X)
            row   = dict(base, backend="catboost", **stats)
            rows.append(row)
            print(f"  catboost  mean={stats['mean_us']:.1f}µs  p99={stats['p99_us']:.1f}µs"
                  f"  reps={stats['n_reps']:,}")

            # --- ONNX Runtime ---
            try:
                stats = bench_onnx(model, X, tmp_dir)
                rows.append(dict(base, backend="onnx", **stats))
                print(f"  onnx      mean={stats['mean_us']:.1f}µs  p99={stats['p99_us']:.1f}µs"
                      f"  reps={stats['n_reps']:,}")
            except Exception as e:
                print(f"  onnx      FAILED: {e}")

            # --- C++ standalone ---
            stats = bench_cpp(model, entry["model_path"], X, args.cache_dir)
            if stats:
                rows.append(dict(base, backend="cpp", **stats))
                print(f"  cpp       mean={stats['mean_us']:.2f}µs  p99={stats['p99_us']:.2f}µs"
                      f"  reps={stats['n_reps']:,}")
            else:
                print("  cpp       FAILED (compile error)")

    out_path = os.path.join(args.out_dir, f"backends_{args.gran}.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nResults → {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
measure_inference_latency.py
============================
Single-sample inference latency for trained WORKLOAD-FORECASTING models across
the three target platforms and both feature sets:

  platform      : arm_server | x86_pcore | x86_ecore
  feature_set   : all (full counter set) | top4
  model_type    : dt | mlp | lstm | transformer
  swept over    : forecast horizon, timesteps

Backends
  DT (sklearn tree / MultiOutputRegressor):  sklearn  · ONNX Runtime
  MLP/LSTM/Transformer (keras):               keras    · ONNX Runtime

Overhead removal (matches cross_platform_prediction/latency):
  - single thread (env vars set by run_inference.sh; onnxruntime pinned in-session)
  - CPU affinity pinned to one core        (--cpu)
  - GC disabled during the hot loop
  - time.perf_counter_ns()                 (nanosecond resolution)
  - adaptive rep count                     (more reps for fast models)

Latency is determined by model ARCHITECTURE (input = timesteps x counters, output
= horizon), not by the frequency the model was trained at, so the frequency
dimension is collapsed: one representative model per
(platform, feature_set, model_type, horizon, timesteps).

Accuracy for the pareto is read from the sweep logs (no condensed CSVs required):
  results/forecasting/logs_10M/{variant}/{freq}GHz/horizon_{h}/timesteps_{t}/{workload}_{model}.log

Outputs
  results/forecasting/latency/inference_latency_{gran}.csv
  results/forecasting/latency/pareto_{gran}.csv

NOTE: benchmarked on THIS host pinned to one core; the `platform` label identifies
which trained model set is being timed, not the native hardware. arm/e-core numbers
are therefore a portability approximation from this host.

Usage
  python3 measure_inference_latency.py
  python3 measure_inference_latency.py --gran 10M --cpu 0
  python3 measure_inference_latency.py --platform x86_pcore --feature_set top4
  python3 measure_inference_latency.py --reps 20000     # fixed rep count
"""

import argparse
import gc
import glob
import os
import re
import shutil
import tempfile
import time

import numpy as np
import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../.."))

# Adaptive rep targets: aim for this many µs of total timing per (model, backend).
TARGET_TIMING_US = 200_000
WARMUP_REPS      = 100
MIN_REPS         = 2_000
MAX_REPS         = 200_000

# variant directory name -> (platform, feature_set). Anything not listed
# (e.g. the *_het_* variants) is ignored.
VARIANT_MAP = {
    "arm_server":                             ("arm_server", "all"),
    "arm_server_top4":                        ("arm_server", "top4"),
    "x86_desktop_heterogeneous_cpu0":         ("x86_pcore",  "all"),
    "x86_desktop_heterogeneous_cpu0_top4":    ("x86_pcore",  "top4"),
    "x86_desktop_heterogeneous_cpu16":        ("x86_ecore",  "all"),
    "x86_desktop_heterogeneous_cpu16_top4":   ("x86_ecore",  "top4"),
}

_MAPE_RE = re.compile(r"'mape':\s*(?:np\.float64\()?([-+0-9.eE]+)")


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_models(models_root, platform_filter=None, feature_set_filter=None):
    """
    One representative model per (platform, feature_set, model_type, horizon,
    timesteps) -- the frequency dimension is collapsed (first freq alphabetically).
    Handles DT as .pkl or .joblib and NN as .keras.
    """
    files = (glob.glob(f"{models_root}/**/*.pkl",    recursive=True) +
             glob.glob(f"{models_root}/**/*.joblib", recursive=True) +
             glob.glob(f"{models_root}/**/*.keras",  recursive=True))
    root_depth = len(models_root.rstrip(os.sep).split(os.sep))
    groups = {}
    for path in sorted(files):
        m = re.search(r'_(dt|mlp|lstm|transformer)\.(pkl|joblib|keras)$',
                      os.path.basename(path))
        if not m:
            continue
        model_type = m.group(1)
        rel = path.split(os.sep)[root_depth:]
        variant = rel[0]
        if variant not in VARIANT_MAP:
            continue
        platform, feature_set = VARIANT_MAP[variant]
        if platform_filter and platform != platform_filter:
            continue
        if feature_set_filter and feature_set != feature_set_filter:
            continue
        try:
            freq      = next(p for p in rel if re.match(r'[\d.]+GHz$', p)).replace('GHz', '')
            horizon   = next(p for p in rel if p.startswith('horizon_')).replace('horizon_', '')
            timesteps = next(p for p in rel if p.startswith('timesteps_')).replace('timesteps_', '')
        except StopIteration:
            continue
        # collapse frequency: keep first model seen per architecture group
        key = (platform, feature_set, model_type, int(horizon), int(timesteps))
        groups.setdefault(key, {"variant": variant, "freq": freq, "path": path})

    return [
        {"platform": k[0], "feature_set": k[1], "model_type": k[2],
         "horizon": k[3], "timesteps": k[4],
         "variant": v["variant"], "freq": v["freq"], "path": v["path"]}
        for k, v in sorted(groups.items())
    ]


# ---------------------------------------------------------------------------
# Model loading + input construction
# ---------------------------------------------------------------------------

def load_sklearn(path):
    import joblib
    return joblib.load(path)


def load_keras(path):
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import keras
    return keras.models.load_model(path)


def sklearn_n_features(model):
    from sklearn.multioutput import MultiOutputRegressor
    return (model.estimators_[0].n_features_in_
            if isinstance(model, MultiOutputRegressor)
            else model.n_features_in_)


def make_input_keras(model):
    shape = model.input_shape
    if len(shape) == 2:
        return np.ones((1, shape[1]), dtype=np.float32)
    return np.ones((1, shape[1], shape[2]), dtype=np.float32)


def n_counters_keras(model, timesteps):
    shape = model.input_shape
    return shape[2] if len(shape) == 3 else shape[1] // timesteps


# ---------------------------------------------------------------------------
# Timing (perf_counter_ns, GC-disabled, adaptive)
# ---------------------------------------------------------------------------

def pin_cpu(cpu_id):
    try:
        os.sched_setaffinity(0, {cpu_id})
        print(f"Pinned to CPU {cpu_id}")
    except (AttributeError, OSError) as e:
        print(f"[WARN] CPU pin failed: {e}")


def adaptive_reps(fn, warmup):
    for _ in range(max(warmup // 5, 1)):
        fn()
    t0 = time.perf_counter_ns()
    for _ in range(1000):
        fn()
    mean_us = (time.perf_counter_ns() - t0) / 1e3 / 1000
    reps = int(TARGET_TIMING_US / max(mean_us, 0.01))
    return max(MIN_REPS, min(MAX_REPS, reps))


def time_fn(fn, reps, warmup):
    for _ in range(warmup):
        fn()
    gc.disable()
    try:
        buf = np.empty(reps, dtype=np.int64)
        for i in range(reps):
            t0 = time.perf_counter_ns()
            fn()
            buf[i] = time.perf_counter_ns() - t0
    finally:
        gc.enable()
    return buf / 1e3   # → µs


def summarize(ts):
    return {
        "latency_mean_us":   float(np.mean(ts)),
        "latency_std_us":    float(np.std(ts)),
        "latency_p50_us":    float(np.percentile(ts, 50)),
        "latency_p90_us":    float(np.percentile(ts, 90)),
        "latency_p95_us":    float(np.percentile(ts, 95)),
        "latency_p99_us":    float(np.percentile(ts, 99)),
        "latency_p999_us":   float(np.percentile(ts, 99.9)),
        "latency_min_us":    float(np.min(ts)),
        "latency_max_us":    float(np.max(ts)),
    }


def null_stats():
    return {k: None for k in ("latency_mean_us", "latency_std_us", "latency_p50_us",
                              "latency_p90_us", "latency_p95_us", "latency_p99_us",
                              "latency_p999_us", "latency_min_us", "latency_max_us")}


def _run(fn, fixed_reps, warmup):
    reps = fixed_reps if fixed_reps > 0 else adaptive_reps(fn, warmup)
    ts   = time_fn(fn, reps, warmup)
    return dict(summarize(ts), n_reps=reps)


# ---------------------------------------------------------------------------
# ONNX / TreeLite conversion (reused from the arm_server version)
# ---------------------------------------------------------------------------

def _ort_session(path):
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.log_severity_level   = 3
    return ort.InferenceSession(path, sess_options=opts,
                                providers=["CPUExecutionProvider"])


def convert_dt_to_onnx(model, n_features, path):
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    proto = convert_sklearn(model,
                            initial_types=[("X", FloatTensorType([None, n_features]))])
    with open(path, "wb") as f:
        f.write(proto.SerializeToString())


def convert_keras_to_onnx(model, X, path):
    import tf2onnx
    import tensorflow as tf
    spec = (tf.TensorSpec([None] + list(X.shape[1:]), tf.float32, name="input"),)
    tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=path)


def onnx_call(onnx_path, X):
    sess = _ort_session(onnx_path)
    name = sess.get_inputs()[0].name
    Xc   = X.astype(np.float32)
    return lambda: sess.run(None, {name: Xc})


# ---------------------------------------------------------------------------
# Per-entry benchmarking
# ---------------------------------------------------------------------------

def bench_dt(entry, reps, warmup, tmp_dir):
    model  = load_sklearn(entry["path"])
    n_feat = sklearn_n_features(model)
    n_cnt  = n_feat // entry["timesteps"]
    X      = np.ones((1, n_feat), dtype=np.float32)
    n_trees = getattr(model, "n_estimators", None)

    out = []
    out.append(("sklearn", n_cnt, n_feat, n_trees,
                _run(lambda: model.predict(X), reps, warmup)))
    try:
        onnx_path = os.path.join(tmp_dir, "dt.onnx")
        convert_dt_to_onnx(model, n_feat, onnx_path)
        out.append(("onnx", n_cnt, n_feat, n_trees,
                    _run(onnx_call(onnx_path, X), reps, warmup)))
    except Exception as e:
        out.append(("onnx", n_cnt, n_feat, n_trees, {**null_stats(), "_error": str(e)}))
    return out


def bench_keras(entry, reps, warmup, tmp_dir):
    model = load_keras(entry["path"])
    X     = make_input_keras(model)
    n_cnt = n_counters_keras(model, entry["timesteps"])
    n_feat = int(np.prod(X.shape[1:]))

    out = []
    out.append(("keras", n_cnt, n_feat, None,
                _run(lambda: model.predict(X, verbose=0), reps, warmup)))
    try:
        onnx_path = os.path.join(tmp_dir, f"{entry['model_type']}.onnx")
        convert_keras_to_onnx(model, X, onnx_path)
        out.append(("onnx", n_cnt, n_feat, None,
                    _run(onnx_call(onnx_path, X), reps, warmup)))
    except Exception as e:
        out.append(("onnx", n_cnt, n_feat, None, {**null_stats(), "_error": str(e)}))
    return out


# ---------------------------------------------------------------------------
# Accuracy (parsed from sweep logs) + pareto
# ---------------------------------------------------------------------------

def mean_mape_from_logs(logs_root, variant, model_type, horizon, timesteps):
    """Mean MAPE over all frequencies & workloads for this architecture group."""
    pattern = os.path.join(
        logs_root, variant, "*GHz", f"horizon_{horizon}",
        f"timesteps_{timesteps}", f"*_{model_type}.log",
    )
    vals = []
    for log in glob.glob(pattern):
        try:
            m = _MAPE_RE.search(open(log, errors="ignore").read())
        except OSError:
            continue
        if m:
            vals.append(float(m.group(1)))
    return float(np.mean(vals)) if vals else float("nan")


def build_pareto(latency_df, logs_root, gran, out_dir):
    native = {"dt": "sklearn", "mlp": "keras", "lstm": "keras", "transformer": "keras"}
    rows = []
    seen = set()
    for _, r in latency_df.iterrows():
        key = (r["platform"], r["feature_set"], r["model_type"], r["horizon"], r["timesteps"])
        if key in seen or r["backend"] != native[r["model_type"]]:
            continue
        seen.add(key)
        rows.append({
            "platform":        r["platform"],
            "feature_set":     r["feature_set"],
            "model_type":      r["model_type"],
            "horizon":         r["horizon"],
            "timesteps":       r["timesteps"],
            "n_counters":      r["n_counters"],
            "mape_mean":       mean_mape_from_logs(logs_root, r["variant"], r["model_type"],
                                                   r["horizon"], r["timesteps"]),
            "latency_mean_us": r["latency_mean_us"],
            "latency_p99_us":  r["latency_p99_us"],
        })
    pareto = pd.DataFrame(rows)
    out_path = os.path.join(out_dir, f"pareto_{gran}.csv")
    pareto.to_csv(out_path, index=False)
    print(f"\nPareto table → {out_path}")
    if len(pareto):
        print(pareto.round(3).to_string(index=False))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gran", default="10M",
                    help="Instruction-window granularity (default: 10M).")
    ap.add_argument("--models_root", default=None,
                    help="Override models root (default: results/forecasting/models_{gran}).")
    ap.add_argument("--logs_root", default=None,
                    help="Override logs root (default: results/forecasting/logs_{gran}).")
    ap.add_argument("--out_dir", default=None,
                    help="Override output dir (default: results/forecasting/latency).")
    ap.add_argument("--platform", default=None,
                    choices=["arm_server", "x86_pcore", "x86_ecore"],
                    help="Restrict to one platform (default: all).")
    ap.add_argument("--feature_set", default=None, choices=["all", "top4"],
                    help="Restrict to one feature set (default: both).")
    ap.add_argument("--reps", type=int, default=0,
                    help="Fixed rep count. 0 = adaptive (default).")
    ap.add_argument("--warmup", type=int, default=WARMUP_REPS)
    ap.add_argument("--cpu", type=int, default=0, help="CPU core to pin to (default: 0).")
    args = ap.parse_args()

    models_root = args.models_root or os.path.join(REPO_ROOT, "results/forecasting", f"models_{args.gran}")
    logs_root   = args.logs_root   or os.path.join(REPO_ROOT, "results/forecasting", f"logs_{args.gran}")
    out_dir     = args.out_dir     or os.path.join(REPO_ROOT, "results/forecasting/latency")

    pin_cpu(args.cpu)

    entries = discover_models(models_root, args.platform, args.feature_set)
    if not entries:
        print(f"No models found under {models_root}"
              + (f" for platform={args.platform}" if args.platform else "")
              + (f" feature_set={args.feature_set}" if args.feature_set else ""))
        return
    print(f"\nFound {len(entries)} (platform × feature_set × model_type × horizon × timesteps) "
          f"groups at {args.gran}\n")

    os.makedirs(out_dir, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(prefix="wf_latency_")
    rows = []
    try:
        for e in entries:
            label = (f"{e['platform']}/{e['feature_set']} | {e['model_type']} | "
                     f"H:{e['horizon']} | T:{e['timesteps']}")
            print(f"[{label}]  {os.path.basename(e['path'])}")
            try:
                results = (bench_dt if e["model_type"] == "dt" else bench_keras)(
                    e, args.reps, args.warmup, tmp_dir)
            except Exception as ex:
                print(f"  LOAD ERROR: {ex}")
                continue
            for backend, n_cnt, n_feat, n_trees, stats in results:
                err = stats.pop("_error", None)
                if err:
                    print(f"  [{backend:9s}] ERROR: {err}")
                else:
                    print(f"  [{backend:9s}] n_cnt={n_cnt} feat={n_feat} "
                          f"mean={stats['latency_mean_us']:7.2f}µs "
                          f"p99={stats['latency_p99_us']:7.2f}µs reps={stats['n_reps']:,}")
                rows.append({
                    "platform": e["platform"], "feature_set": e["feature_set"],
                    "model_type": e["model_type"], "horizon": e["horizon"],
                    "timesteps": e["timesteps"], "n_counters": n_cnt,
                    "n_features": n_feat, "n_trees": n_trees, "backend": backend,
                    "variant": e["variant"], "freq": e["freq"], "model_path": e["path"],
                    **stats,
                })
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    df = pd.DataFrame(rows)
    out_path = os.path.join(out_dir, f"inference_latency_{args.gran}.csv")
    df.to_csv(out_path, index=False)
    print(f"\nResults → {out_path}")

    build_pareto(df, logs_root, args.gran, out_dir)


if __name__ == "__main__":
    main()

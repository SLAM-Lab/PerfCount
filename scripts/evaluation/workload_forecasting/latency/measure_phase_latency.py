#!/usr/bin/env python3
"""
measure_phase_latency.py
========================
Phase-UNAWARE vs. phase-AWARE single-sample forecasting latency on the x86
heterogeneous desktop (P-core cpu0, E-core cpu16) only.

  phase-unaware : one global forecaster           ->  global.predict(window)
  phase-aware   : classify the window, then route ->  gmm.predict(row)         [phase id]
                                                    + phase_k.predict(window)   [routed model]

The routed per-phase model shares the global model's architecture, so its inference
cost equals the phase-unaware cost. The only added work is the phase classification
(a Gaussian mixture over the per-window counter vector) plus an O(1) model lookup.
This benchmark isolates that overhead and reports whether phase-aware forecasting
stays within a scheduling interval.

It times the ACTUAL ensembles saved by the phase-forecasting study
(results/forecasting/phase_forecasting/models/<bench>/<model>_gmm/), which are the
stateless top4b models (five-window history, four counters) the study reports.
Latency is architecture-driven, so it is averaged over a sample of benchmarks per
(core, model) to smooth per-tree size variation. Timing machinery (single-thread,
CPU-pinned, GC-disabled, perf_counter_ns, adaptive reps, ONNX export) is reused from
measure_inference_latency.py.

Outputs
  results/forecasting/latency/phase_latency_{gran}.csv
    core, model_type, backend, n_benches,
    unaware_mean_us, unaware_p99_us, classify_mean_us,
    aware_mean_us, aware_p99_us, overhead_pct
"""
import argparse
import glob
import json
import os
import shutil
import tempfile

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

import measure_inference_latency as base

REPO_ROOT = base.REPO_ROOT
ENSEMBLE_ROOT = os.path.join(REPO_ROOT, "results/forecasting/phase_forecasting/models")
PHASE_COUNT = 6
CORES = [("cpu0", "P-core"), ("cpu16", "E-core")]
MODELS = os.environ.get("PHASE_LAT_MODELS", "dt mlp lstm transformer").split()
NATIVE = {"dt": "sklearn", "mlp": "keras", "lstm": "keras", "transformer": "keras"}


ENS_SUFFIX = os.environ.get("PHASE_LAT_SUFFIX", "_gmm_delta")


def ensembles_for(core, model, n_benches):
    """Return up to n_benches ensemble dirs (absolute) for this core+model.
    Defaults to the residual-over-persistence (delta) ensembles the study saves,
    matching <model>_gmm_delta exactly (not the _hetinfer variants)."""
    pat = os.path.join(ENSEMBLE_ROOT, f"aligned_*_{core}", f"{model}{ENS_SUFFIX}")
    dirs = sorted(d for d in glob.glob(pat) if os.path.isfile(os.path.join(d, "metadata.json")))
    return dirs[:n_benches]


def load_pair(ens_dir, model):
    """Load (global, phase_k, timesteps, n_counters) from an ensemble dir.
    phase_k is a per-phase model (same architecture) or falls back to global."""
    meta = json.load(open(os.path.join(ens_dir, "metadata.json")))
    ts = int(meta.get("timesteps", 5))
    ext = "joblib" if model == "dt" else "keras"
    load = base.load_sklearn if model == "dt" else base.load_keras
    g = load(os.path.join(ens_dir, f"global.{ext}"))
    phase_files = sorted(glob.glob(os.path.join(ens_dir, f"phase*.{ext}")))
    p = load(phase_files[0]) if phase_files else g
    if model == "dt":
        n_feat = base.sklearn_n_features(g)
        n_cnt = n_feat // ts
    else:
        n_cnt = base.n_counters_keras(g, ts)
    return g, p, ts, n_cnt


def make_input(model, mdl, ts, n_cnt):
    if model == "dt":
        return np.ones((1, n_cnt * ts), dtype=np.float32)
    return base.make_input_keras(mdl)


def predict_call(model, mdl, X, backend, tmp_dir, tag):
    if backend == "native":
        if model == "dt":
            return lambda: mdl.predict(X)
        return lambda: mdl.predict(X, verbose=0)
    path = os.path.join(tmp_dir, f"{tag}.onnx")
    if model == "dt":
        base.convert_dt_to_onnx(mdl, X.shape[1], path)
    else:
        base.convert_keras_to_onnx(mdl, X, path)
    return base.onnx_call(path, X)


def classify_call(gmm, row, n_cnt, backend, tmp_dir, tag):
    """Phase classification (GMM predict) in the SAME runtime as the forecaster.
    Under native, sklearn's predict() is dominated by Python call overhead
    (~130 us for a 5-component 4-D mixture whose math is ~1 us), so with
    compiled forecasters the classifier is compiled too for a fair comparison."""
    if backend == "native":
        return lambda: gmm.predict(row)
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    path = os.path.join(tmp_dir, f"{tag}.onnx")
    onx = convert_sklearn(gmm, initial_types=[("X", FloatTensorType([None, n_cnt]))])
    with open(path, "wb") as f:
        f.write(onx.SerializeToString())
    return base.onnx_call(path, row.astype(np.float32))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gran", default="10M")
    ap.add_argument("--reps", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=base.WARMUP_REPS)
    ap.add_argument("--cpu", type=int, default=0)
    ap.add_argument("--n_benches", type=int, default=6,
                    help="benchmarks to average per (core, model).")
    ap.add_argument("--backends", default="native,onnx")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(REPO_ROOT, "results/forecasting/latency")
    os.makedirs(out_dir, exist_ok=True)
    base.pin_cpu(args.cpu)
    rng = np.random.default_rng(0)
    backends = [b for b in args.backends.split(",") if b]

    tmp_dir = tempfile.mkdtemp(prefix="phase_latency_")
    rows = []
    try:
        for core, core_label in CORES:
            for model in MODELS:
                dirs = ensembles_for(core, model, args.n_benches)
                if not dirs:
                    print(f"  [{core_label}/{model}] no ensembles, skipping")
                    continue
                for backend in backends:
                    un_m, un_p, cl_m, aw_m, aw_p, ncnt = [], [], [], [], [], None
                    for d in dirs:
                        try:
                            g, p, ts, n_cnt = load_pair(d, model)
                            X = make_input(model, g, ts, n_cnt)
                            gmm = GaussianMixture(
                                n_components=min(PHASE_COUNT, max(2, n_cnt + 1)),
                                covariance_type="full", random_state=0
                            ).fit(rng.normal(size=(400, n_cnt)))
                            row = np.ones((1, n_cnt), dtype=np.float64)
                            fwd_g = predict_call(model, g, X, backend, tmp_dir, "g")
                            fwd_p = predict_call(model, p, X, backend, tmp_dir, "p")
                            clf = classify_call(gmm, row, n_cnt, backend, tmp_dir, "gmm")
                            aware = lambda: (clf(), fwd_p())
                            un = base._run(fwd_g, args.reps, args.warmup)
                            cl = base._run(clf, args.reps, args.warmup)
                            aw = base._run(aware, args.reps, args.warmup)
                        except Exception as ex:
                            print(f"  [{core_label}/{model}/{backend}] {os.path.basename(os.path.dirname(d))}: {ex}")
                            continue
                        un_m.append(un["latency_mean_us"]); un_p.append(un["latency_p99_us"])
                        cl_m.append(cl["latency_mean_us"])
                        aw_m.append(aw["latency_mean_us"]); aw_p.append(aw["latency_p99_us"])
                        ncnt = n_cnt
                    if not un_m:
                        continue
                    um, ap_, cm, am, apk = (np.mean(un_m), np.mean(un_p), np.mean(cl_m),
                                            np.mean(aw_m), np.mean(aw_p))
                    overhead = 100.0 * (am - um) / um
                    rows.append({
                        "core": core, "core_label": core_label, "model_type": model,
                        "backend": backend, "n_benches": len(un_m), "n_counters": ncnt,
                        "unaware_mean_us": um, "unaware_p99_us": ap_, "classify_mean_us": cm,
                        "aware_mean_us": am, "aware_p99_us": apk, "overhead_pct": overhead,
                    })
                    print(f"  [{core_label}/{model}/{backend}] n={len(un_m)} "
                          f"unaware={um:.2f}us classify={cm:.2f}us aware={am:.2f}us (+{overhead:.1f}%)")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    df = pd.DataFrame(rows)
    out_path = os.path.join(out_dir, f"phase_latency_{args.gran}.csv")
    df.to_csv(out_path, index=False)
    print(f"\nResults -> {out_path}")
    if len(df):
        print(df.round(2).to_string(index=False))


if __name__ == "__main__":
    main()

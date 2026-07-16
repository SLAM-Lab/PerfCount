#!/usr/bin/env python3
"""
predict_cross_config.py
=======================
Cross-config workload forecasting: predict a benchmark's future `ref_cycles` at a
TARGET config C = (core_t, freq_t) given history observed at a SOURCE config
S = (core_s, freq_s), by translate-then-forecast:

  1. take S's observed window,
  2. translate `ref_cycles` into C's space with the cross-platform model
     (cross-processor when cores differ, cross-frequency when only freq differs;
     identity when S==C), COPYING the other top-4 counters unchanged,
  3. run C's own pre-trained per-config forecaster,
  4. compare to C's aligned ground-truth future.

Methods compared per (bench, S, C, model, H, T):
  translated  - proposed (ref_cycles translated, others copied)
  oracle      - C's forecaster on C's own history (upper bound)
  naive       - C's forecaster on S's untranslated history
  persistence - carry the last (translated) ref_cycles forward

All error is MAPE against C's original `ref_cycles`, using the exact metric the
forecasting sweep reports (src/evaluate.get_regression_metrics).

Reuses the trained artifacts as-is (no new training):
  forecasters  results/forecasting/models_10M/x86_desktop_heterogeneous_cpu{c}_top4/...
  translators  results/cross_platform/cross_{proc,freq}/x86_10M/...

Usage:
  python predict_cross_config.py --benchmark dacapo_avrora --model dt --horizon 1 --timesteps 5
  python predict_cross_config.py --benchmark spec_505.mcf_r --model dt --src 0:4.0 --tgt 16:4.0
"""
import os
import re
import sys
import glob
import argparse
import itertools
from types import SimpleNamespace

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WF_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
REPO_ROOT = os.path.abspath(os.path.join(WF_DIR, "..", "..", ".."))
sys.path.insert(0, WF_DIR)

from src import create_dataset as cds     # noqa: E402
from src import evaluate as ev            # noqa: E402
from src.Predictor import PredictorInputs  # noqa: E402

DATASET   = "x86_desktop_heterogeneous"
TARGET    = "ref_cycles"
CPUS      = [0, 16]
FREQS     = [1.0, 2.0, 3.0, 4.0]

# Forecaster input set per variant (first counter is the forecast target).
# top4b swaps the config-dependent branch_misses for the config-invariant branches.
VARIANT_COUNTERS = {
    "top4":  ["ref_cycles", "cpu_cycles", "branch_misses", "instructions"],
    "top4b": ["ref_cycles", "cpu_cycles", "branches", "instructions"],
}

MODELS_ROOT = os.path.join(REPO_ROOT, "results/forecasting/models_10M")
CBM_PROC    = os.path.join(REPO_ROOT, "results/cross_platform/cross_proc/x86_10M")
CBM_FREQ    = os.path.join(REPO_ROOT, "results/cross_platform/cross_freq/x86_10M")
# Per-counter (non-ref_cycles) cross-processor translators, e.g. cpu_cycles.
CBM_PROC_CT = os.path.join(CBM_PROC, "counter_translation")
OUT_DIR     = os.path.join(REPO_ROOT, "results/forecasting/cross_config")
DATA_DIR    = os.path.join(REPO_ROOT, "processed_data_10M", DATASET)


# ---------------------------------------------------------------------------
# args / trace loading
# ---------------------------------------------------------------------------

def build_args(model, timesteps, horizon, counters):
    """Namespace matching the top4 sweep's forecasting config."""
    return SimpleNamespace(
        input_counters=list(counters), model=model, timesteps=timesteps,
        forecast_horizon=horizon, scaler="minmax", filter="none", filter_size=3,
        pca=None, train_size=70, batch_size=32, start_drop_count=0, end_drop_count=0,
        heterogeneous_prob=0.0, add_heterogeneity_features=False,
    )


def load_full_trace(workload):
    """All-column phase-concatenated trace for aligned_{rest}_{freq}GHz_cpu{cpu}.
    Dedup by phase number keeping the alphabetically-first path (dacapo_c1 < c2)."""
    matches = glob.glob(os.path.join(DATA_DIR, "**", f"{workload}.csv"), recursive=True)
    if matches:
        return pd.read_csv(matches[0])
    phase_paths = glob.glob(os.path.join(DATA_DIR, "**", f"{workload}_phase*.csv"), recursive=True)
    by_num = {}
    for p in sorted(phase_paths):
        n = int(re.search(r"_phase(\d+)\.csv$", p).group(1))
        by_num.setdefault(n, p)
    if not by_num:
        return None
    dfs = [pd.read_csv(by_num[n]) for n in sorted(by_num)]
    common = list(set.intersection(*[set(d.columns) for d in dfs]))
    return pd.concat([d[common] for d in dfs], ignore_index=True)


def workload_name(rest, freq, cpu):
    return f"aligned_{rest}_{freq}GHz_cpu{cpu}"


# ---------------------------------------------------------------------------
# translation (ref_cycles only; other counters copied)
# ---------------------------------------------------------------------------

def _ct_suite(rest):
    """Suite label used in the counter_translation tree (spec2017/spec2026/dacapo)."""
    if rest.startswith("dacapo"):
        return "dacapo"
    m = re.match(r"spec_(\d+)", rest)
    if m:
        return "spec2026" if int(m.group(1)) >= 700 else "spec2017"
    return None


def _load_ct(counter, s_cpu, s_freq, t_cpu, t_freq, rest):
    """Load a per-counter cross-proc translator from the counter_translation tree."""
    suite = _ct_suite(rest)
    if suite is None:
        return None
    path = os.path.join(
        CBM_PROC_CT, f"cpu{s_cpu}_to_cpu{t_cpu}", suite, counter, "top4",
        f"cpu{s_cpu}_{s_freq}GHz_to_cpu{t_cpu}_{t_freq}GHz", f"model_{rest}.cbm",
    )
    if not os.path.exists(path):
        return None
    m = cds._CatBoost()
    m.load_model(path)
    return m


def _apply_ratio(cbm, full_df, counter):
    """translated = exp(predicted log-ratio) * source[counter]; int64 array or None."""
    feats = list(cbm.feature_names_)
    if not all(f in full_df.columns for f in feats):
        return None
    src = full_df[counter].values
    mask = src > 0
    pred_log = cbm.predict(full_df[feats])
    return np.where(mask, np.exp(pred_log) * src, src).round().astype(np.int64)


def translate_counter(full_src, rest, counter, s_cpu, s_freq, t_cpu, t_freq):
    """Translated array for `counter` S->C, or None if no translator. Identity when
    S==C. ref_cycles uses the top-level cross_proc/cross_freq tree; other counters
    (e.g. cpu_cycles) use the counter_translation tree and translate cross-core only
    (they are frequency-invariant, so same-core freq changes copy exactly)."""
    if (s_cpu, s_freq) == (t_cpu, t_freq):
        return full_src[counter].values
    if counter == "ref_cycles":
        if s_cpu != t_cpu:
            cbm = cds._load_cbm_cross_proc(CBM_PROC, s_cpu, s_freq, t_cpu, t_freq, rest)
        else:
            cbm = cds._load_cbm(os.path.join(CBM_FREQ, f"cpu{t_cpu}"), s_freq, t_freq, rest)
        if cbm is None:
            return None
        return cds._translate_ref_cycles(cbm, full_src)
    # non-ref counter (cpu_cycles): cross-core only; same-core copy (freq-invariant)
    if s_cpu == t_cpu:
        return full_src[counter].values
    cbm = _load_ct(counter, s_cpu, s_freq, t_cpu, t_freq, rest)
    if cbm is None:
        return None
    return _apply_ratio(cbm, full_src, counter)


# ---------------------------------------------------------------------------
# forecasting
# ---------------------------------------------------------------------------

def load_forecaster(rest, t_cpu, t_freq, model, horizon, timesteps, variant):
    base = os.path.join(
        MODELS_ROOT, f"{DATASET}_cpu{t_cpu}_{variant}", f"{t_freq}GHz",
        f"horizon_{horizon}", f"timesteps_{timesteps}",
        f"{workload_name(rest, t_freq, t_cpu)}_{model}",
    )
    if model == "dt":
        import joblib
        for ext in (".pkl", ".joblib"):
            if os.path.exists(base + ext):
                return joblib.load(base + ext), "sklearn"
        return None, None
    if os.path.exists(base + ".keras"):
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        import keras
        return keras.models.load_model(base + ".keras"), "keras"
    return None, None


# Forecaster cache: load each (rest, tgt, model, h, t, variant) once and reuse across
# all source pairs (the keras load is the dominant cost). Entry also caches an ONNX
# session when --onnx is set.
_FCACHE = {}
_USE_ONNX = False


def get_forecaster(rest, t_cpu, t_freq, model, horizon, timesteps, variant):
    key = (rest, t_cpu, t_freq, model, horizon, timesteps, variant)
    if key not in _FCACHE:
        obj, kind = load_forecaster(rest, t_cpu, t_freq, model, horizon, timesteps, variant)
        _FCACHE[key] = {"model": obj, "kind": kind, "sess": None, "inp": None}
    return _FCACHE[key]


def _ensure_onnx(entry, X):
    """Convert a cached keras forecaster to a (cached) ONNX Runtime session.

    Routes through a SavedModel export rather than tf2onnx.from_keras: the latter
    hits a KeyError on Keras-3 output-tensor name lookup for recurrent models
    (LSTM). Exporting a serving signature and using from_saved_model is robust
    across MLP / LSTM / Transformer alike.
    """
    import tempfile
    import tf2onnx
    import tensorflow as tf
    import onnxruntime as ort
    path = os.path.join(tempfile.gettempdir(), f"ccfc_{id(entry)}.onnx")
    spec = tf.TensorSpec([None] + list(X.shape[1:]), tf.float32, name="input")

    model = entry["model"]

    @tf.function(input_signature=[spec])
    def serve(x):
        return model(x)

    # from_function traces the concrete graph (dynamic batch via the None-dim
    # TensorSpec) and names outputs from graph tensors -- avoiding both the
    # from_keras Keras-3 output-name KeyError and the static-batch LSTM shapes
    # baked in by SavedModel export.
    tf2onnx.convert.from_function(serve, input_signature=[spec], opset=13, output_path=path)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.log_severity_level = 3
    entry["sess"] = ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])
    entry["inp"] = entry["sess"].get_inputs()[0].name


def predict_entry(entry, X):
    kind = entry["kind"]
    if kind == "sklearn":
        return entry["model"].predict(X)
    # Keras. Use ONNX when requested, but fall back to keras-native per entry if
    # conversion/inference fails (e.g. stateful LSTM: fixed batch size can't be
    # exported to a dynamic-batch ONNX graph). Fallback is sticky per entry.
    if _USE_ONNX and entry.get("onnx_state") != "failed":
        try:
            if entry["sess"] is None:
                _ensure_onnx(entry, X)
            return entry["sess"].run(None, {entry["inp"]: X.astype(np.float32)})[0]
        except Exception as e:
            entry["onnx_state"] = "failed"
            entry["sess"] = None
            sys.stderr.write(f"[onnx->keras fallback] {type(e).__name__}: {e}\n")
    return entry["model"].predict(X, verbose=0)


def forecast_mape(args, entry, x_top4, c_top4):
    """MAPE of C's forecaster fed `x_top4` (already in C's space), against C's
    original ground truth. c_top4 supplies C's scalers + ground-truth future."""
    c_ts = cds.get_transformed_time_series(args, c_top4)
    c_ts.set_train_test_split(args)
    train = PredictorInputs(args, c_ts.X_train, c_ts.y_train)

    m_ts = cds.get_transformed_time_series(args, x_top4)
    m_ts.set_train_test_split(args)
    idx = c_ts.X_test.index.intersection(m_ts.X.index)
    X_test = m_ts.X.loc[idx]
    y_test = c_ts.y_test.loc[idx]

    test = PredictorInputs(args, X_test, y_test, train.in_scaler, train.out_scaler,
                           batch_padding=True)
    yhat = np.asarray(predict_entry(entry, test.X))
    test.add_predictions(args, yhat)
    preds = c_ts.invert_predicted_transforms(test.y_hat)

    val = ev.get_validation_set("original", c_ts, TARGET, 1, args.forecast_horizon)
    both = val.loc[preds.index].join(pd.DataFrame(index=preds.index)).dropna()
    preds = preds.loc[both.index]
    return ev.get_regression_metrics(both.values, preds.values, args.forecast_horizon)["mape"]


def persistence_mape(args, x_ref, c_ref):
    """Carry the last (translated) ref_cycles forward H steps; MAPE vs C's future."""
    T, H = args.timesteps, args.forecast_horizon
    n = min(len(x_ref), len(c_ref))
    split = int(n * args.train_size / 100)
    rows = []
    for i in range(max(split, T - 1), n - H):
        true = c_ref[i + H]
        pred = x_ref[i]
        if true != 0:
            rows.append(abs(true - pred) / true * 100.0)
    return float(np.mean(rows)) if rows else float("nan")


# ---------------------------------------------------------------------------
# per-pair evaluation
# ---------------------------------------------------------------------------

def eval_pair(rest, s_cpu, s_freq, t_cpu, t_freq, model, horizon, timesteps, variant, translate):
    counters = VARIANT_COUNTERS[variant]
    args = build_args(model, timesteps, horizon, counters)

    full_s = load_full_trace(workload_name(rest, s_freq, s_cpu))
    full_t = load_full_trace(workload_name(rest, t_freq, t_cpu))
    if full_s is None or full_t is None or TARGET not in full_s or TARGET not in full_t:
        return []
    n = min(len(full_s), len(full_t))
    full_s, full_t = full_s.iloc[:n].reset_index(drop=True), full_t.iloc[:n].reset_index(drop=True)

    entry = get_forecaster(rest, t_cpu, t_freq, model, horizon, timesteps, variant)
    if entry["model"] is None:
        return []

    if not all(c in full_s.columns for c in counters):
        return []
    c_in = full_t[counters].copy()

    tset = [c for c in translate if c in counters]
    rows = []
    base = dict(src_cpu=s_cpu, src_freq=s_freq, tgt_cpu=t_cpu, tgt_freq=t_freq,
                bench=rest, model=model, horizon=horizon, timesteps=timesteps,
                variant=variant, translate="+".join(tset))

    def add(method, mape):
        rows.append(dict(base, method=method, mape=mape))

    # oracle: C's forecaster on C's own history
    add("oracle", forecast_mape(args, entry, c_in, c_in))
    # naive: C's forecaster on S's untranslated counters
    add("naive", forecast_mape(args, entry, full_s[counters].copy(), c_in))
    # translated: translate each counter in `tset` S->C, copy the rest
    x = full_s[counters].copy()
    ok = True
    for c in tset:
        tc = translate_counter(full_s, rest, c, s_cpu, s_freq, t_cpu, t_freq)
        if tc is None:
            ok = False
            break
        x[c] = tc
    if ok:
        add("translated", forecast_mape(args, entry, x, c_in))
        add("persistence", persistence_mape(args, x[TARGET].values, full_t[TARGET].values))
    return rows


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def _parse_cfg(s):
    cpu, freq = s.split(":")
    return int(cpu), float(freq)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", required=True, help="bench 'rest', e.g. dacapo_avrora / spec_505.mcf_r")
    ap.add_argument("--model", default="dt")
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--timesteps", type=int, default=5)
    ap.add_argument("--variant", default="top4", choices=list(VARIANT_COUNTERS),
                    help="forecaster/counter-set variant (top4=branch_misses, top4b=branches)")
    ap.add_argument("--translate", nargs="+", default=["ref_cycles"],
                    help="counters to translate S->C (rest copied). e.g. ref_cycles cpu_cycles")
    ap.add_argument("--src", default=None, help="restrict source config, cpu:freq e.g. 0:4.0")
    ap.add_argument("--tgt", default=None, help="restrict target config, cpu:freq e.g. 16:4.0")
    ap.add_argument("--onnx", action="store_true",
                    help="run keras (MLP/LSTM/Transformer) inference via ONNX Runtime (faster)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    global _USE_ONNX
    _USE_ONNX = args.onnx

    configs = [(c, f) for c in CPUS for f in FREQS]
    srcs = [_parse_cfg(args.src)] if args.src else configs
    tgts = [_parse_cfg(args.tgt)] if args.tgt else configs

    rows = []
    for (sc, sf), (tc, tf) in itertools.product(srcs, tgts):
        r = eval_pair(args.benchmark, sc, sf, tc, tf, args.model, args.horizon, args.timesteps,
                      args.variant, args.translate)
        for row in r:
            print(f"  {sc}@{sf}->{tc}@{tf} [{row['method']:11s}] MAPE={row['mape']:.2f}")
        rows.extend(r)

    if not rows:
        print("No results (missing traces/forecasters/translators).")
        return
    os.makedirs(OUT_DIR, exist_ok=True)
    out = args.out or os.path.join(OUT_DIR, "cross_config_10M.csv")
    df = pd.DataFrame(rows)
    if os.path.exists(out):
        df = pd.concat([pd.read_csv(out), df], ignore_index=True)
    df.to_csv(out, index=False)
    print(f"\nWrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()

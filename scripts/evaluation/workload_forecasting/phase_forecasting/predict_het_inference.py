#!/usr/bin/env python3
"""
predict_het_inference.py  (C4: phase-aware HETEROGENEOUS INFERENCE)
==================================================================
Cold prediction: forecast a TARGET config C's next interval while only observing a
FOREIGN config S (e.g. the workload runs on core/freq S; predict its behavior on C to
decide a migration). Per-phase + delta forecasters are trained on C's own history
(offline profiling); at inference the test INPUT comes from S, either raw (naive) or
translated to C (via the cross-freq / cross-proc CatBoost models). Ground truth is
always C's real future.

The honest baseline here is TRANSLATED-PERSISTENCE: with no clean C observations, the
best "carry-forward" estimate of C[i] is translate(S[i]). (Homogeneous persistence —
carrying C's own value — is unavailable in the cold case and only shown for reference.)

Test-input regimes x forecaster methods, all scored (MAPE) vs C's true future:
  regimes  : oracle (C's own features) | translated (S->C) | naive (raw S)
  methods  : global | per_phase | per_phase_gated
  baselines: translated_persistence | naive_persistence | homog_persistence(ref only)

Reuses predict_phase_forecast (training/gate) + predict_cross_config (translation).
Counter set defaults to top4b (branches; config-invariant except ref_cycles + cpu_cycles).

Usage:
  python predict_het_inference.py --benchmark aligned_spec_505.mcf_r_4.0GHz_cpu0 \
      --source 16:4.0 --model dt --delta --classifier gmm --phase_count 6 --out r.csv
  # --source is the observed foreign config "cpu:freq" (cross-proc if cpu differs,
  #   cross-freq if only freq differs).
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd

WF_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CC_DIR = os.path.join(WF_DIR, "heterogeneous_forecasting", "inference")
if not os.path.isdir(CC_DIR):
    raise RuntimeError(f"cross-config inference dir not found: {CC_DIR}")
for p in (WF_DIR, CC_DIR, os.path.dirname(os.path.abspath(__file__))):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.create_dataset import (get_separate_time_series_splits, get_transformed_time_series,
                                reshape_with_batch_size, BENCHMARK_NAME_NOPHASE_RE)
from src.classify import pre_classification, classify
import src.evaluate as ev
import predict_phase_forecast as pf         # train_model, predict_model, score, gate_phases
import predict_cross_config as cc           # load_full_trace, translate_counter, workload_name
from types import SimpleNamespace

TARGET = "ref_cycles"


def build_args(a):
    return SimpleNamespace(
        input_counters=list(a.counters), model=a.model, timesteps=a.timesteps,
        forecast_horizon=1, scaler="minmax", filter="none", filter_size=a.filter_size,
        pca=None, train_size=70, batch_size=32, start_drop_count=0, end_drop_count=0,
        heterogeneous_prob=0.0, add_heterogeneity_features=False,
        classifier=a.classifier, phase_count=a.phase_count, classifier_threshold=1.0,
        distance_metric="euclidean", W=100, N1=10, multicore_phases=None,
        # model hyperparams
        epochs=a.epochs, neurons=16, stateless=True, dense_hidden_layers=[50, 50],
        early_stopping=False, patience=5, loss_function="mse", regression_activation="linear",
        optimizer="adam", stacked_layers=2, tree_max_depth=3, num_heads=2, dropout_rate=0.2,
        svm_kernel="rbf", svm_regularization=1.0, svm_epsilon=0.1, max_iter=-1,
        min_phase_train=a.min_phase_train, gate_folds=a.gate_folds, no_gate=a.no_gate,
        delta=a.delta, phase_pred_depth=8,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", required=True, help="target config, aligned_<rest>_<f>GHz_cpu<c>")
    ap.add_argument("--source", required=True, help="observed foreign config 'cpu:freq', e.g. 16:4.0")
    ap.add_argument("--dataset", default="x86_desktop_heterogeneous")
    ap.add_argument("--counters", nargs="+",
                    default=["ref_cycles", "cpu_cycles", "branches", "instructions"])
    ap.add_argument("--model", default="dt")
    ap.add_argument("--classifier", default="gmm")
    ap.add_argument("--phase_count", type=int, default=6)
    ap.add_argument("--timesteps", type=int, default=5)
    ap.add_argument("--filter_size", type=int, default=3)
    ap.add_argument("--delta", action="store_true")
    ap.add_argument("--no_gate", action="store_true")
    ap.add_argument("--gate_folds", type=int, default=4)
    ap.add_argument("--min_phase_train", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--save_models_dir", default=None,
                    help="persist the target-config ensemble (global + per-phase + metadata)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    args = build_args(a)

    m = BENCHMARK_NAME_NOPHASE_RE.match(a.benchmark)
    rest, t_freq, t_cpu = m.group("rest"), m.group("freq"), m.group("cpu")
    s_cpu, s_freq = a.source.split(":")
    counters = a.counters

    fullC = cc.load_full_trace(cc.workload_name(rest, t_freq, t_cpu))
    fullS = cc.load_full_trace(cc.workload_name(rest, s_freq, s_cpu))
    if fullC is None or fullS is None:
        print(f"[skip] missing trace for {rest} C={t_cpu}:{t_freq} or S={s_cpu}:{s_freq}")
        return
    n = min(len(fullC), len(fullS))
    fullC, fullS = fullC.iloc[:n].reset_index(drop=True), fullS.iloc[:n].reset_index(drop=True)
    if not all(c in fullC.columns and c in fullS.columns for c in counters):
        print(f"[skip] missing counters for {rest}")
        return

    dataC = fullC[counters].copy()

    # phases on C (workload phase structure) ------------------------------
    phases = classify(args, pre_classification(dataC, args.filter_size)).astype(int)
    phases.index = dataC.index

    # train per-phase + global models on C's history ----------------------
    X_tr, y_tr, X_te, y_te, train_ts, test_ts = get_separate_time_series_splits(args, dataC)
    X_tr, y_tr, X_te, y_te = reshape_with_batch_size(X_tr, y_tr, X_te, y_te, args.batch_size)
    ph_tr = phases.reindex(X_tr.index)
    anchor_col = f"t_{TARGET}" if f"t_{TARGET}" in X_tr.columns else TARGET

    a_tr = X_tr[anchor_col]
    if args.delta:
        y_tr = y_tr.sub(a_tr.values, axis=0)

    g_model = pf.train_model(args, X_tr, y_tr)
    phase_models = {}
    for p in sorted(ph_tr.dropna().unique()):
        if (ph_tr == p).sum() < args.min_phase_train:
            continue
        try:
            phase_models[int(p)] = pf.train_model(args, X_tr[(ph_tr == p).values],
                                                   y_tr[(ph_tr == p).values], g_model[1], g_model[2])
        except Exception:
            pass
    covered = sorted(phase_models)
    accepted = covered if args.no_gate else pf.gate_phases(args, X_tr, y_tr, ph_tr, covered, test_ts)

    # C ground truth + transition mask on the test region -----------------
    val_all = ev.get_validation_set("original", test_ts, TARGET, 1, 1)
    c_test_idx = X_te.index
    nphase = phases.shift(-1).reindex(c_test_idx)
    is_trans = (phases.reindex(c_test_idx) != nphase) & nphase.notna()

    # translated + naive S frames -----------------------------------------
    fS_naive = fullS[counters].copy()
    fS_trans = fullS[counters].copy()
    tr_ref = cc.translate_counter(fullS, rest, "ref_cycles", s_cpu, s_freq, t_cpu, t_freq)
    if tr_ref is not None:
        fS_trans["ref_cycles"] = tr_ref
    if s_cpu != t_cpu and "cpu_cycles" in counters:
        tr_cpu = cc.translate_counter(fullS, rest, "cpu_cycles", s_cpu, s_freq, t_cpu, t_freq)
        if tr_cpu is not None:
            fS_trans["cpu_cycles"] = tr_cpu

    rows = []
    base = dict(bench=a.benchmark, source=a.source, model=a.model, classifier=a.classifier,
                mode=("cross_proc" if s_cpu != t_cpu else "cross_freq"),
                delta=int(args.delta), phase_count=int(phases.nunique()))

    def add(regime, method, phase, mape, nrow):
        rows.append(dict(base, regime=regime, method=method, phase=phase, mape=mape, n=nrow))

    def score_split(pred_abs, tag_regime, tag_method):
        mape, ape = pf.score(val_all.loc[c_test_idx], pred_abs)
        add(tag_regime, tag_method, "all", mape, len(ape))
        tm = is_trans.reindex(ape.index).fillna(False).values
        if tm.sum():
            add(tag_regime, tag_method, "transition", float(ape[tm].mean()), int(tm.sum()))
        return mape

    def eval_regime(frame, regime):
        ts = get_transformed_time_series(args, frame)
        ts.set_train_test_split(args)
        idx = c_test_idx.intersection(ts.X.index)
        Xte = ts.X.loc[idx]
        yte = y_te.loc[idx] if args.delta else y_te.loc[idx]  # only used for shape/index
        anchor = Xte[anchor_col]
        ph = phases.reindex(idx)

        def recon(pdf):
            return pdf.add(anchor.reindex(pdf.index), axis=0) if args.delta else pdf

        g_pred = recon(pf.predict_model(args, g_model[0], Xte, yte, g_model[1], g_model[2], ts))

        def route(allowed):
            pred = g_pred.copy()
            for p in allowed:
                if p not in phase_models:
                    continue
                sub = ph.index[ph == p]
                if len(sub) == 0:
                    continue
                pp = recon(pf.predict_model(args, phase_models[p][0], Xte.loc[sub], yte.loc[sub],
                                            phase_models[p][1], phase_models[p][2], ts))
                pred.loc[pp.index] = pp
            return pred.sort_index()

        score_split(g_pred, regime, "global")
        score_split(route(covered), regime, "per_phase")
        score_split(route(accepted), regime, "per_phase_gated")

        # persistence for this regime = carry this frame's observed value forward
        persist = pd.DataFrame({TARGET: anchor.values}, index=idx)
        score_split(persist, regime, "persistence")

    eval_regime(dataC, "oracle")
    eval_regime(fS_trans, "translated")
    eval_regime(fS_naive, "naive")

    # report ---------------------------------------------------------------
    print(f"\n{a.benchmark}  S={a.source}  [{a.model}/{base['mode']}/"
          f"{'delta' if args.delta else 'abs'}]  covered={covered} accepted={accepted}")
    piv = pd.DataFrame(rows)
    w = piv[piv.phase == "all"].pivot_table(index="regime", columns="method", values="mape")
    order = [r for r in ["oracle", "translated", "naive"] if r in w.index]
    print(f"  {'regime':12s} {'global':>8} {'per_phase':>10} {'gated':>8} {'persist':>8}")
    for r in order:
        print(f"  {r:12s} " + " ".join(
            f"{w.loc[r, c]:8.2f}" if c in w.columns and pd.notna(w.loc[r, c]) else f"{'-':>8}"
            for c in ["global", "per_phase", "per_phase_gated", "persistence"]))

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        pd.DataFrame(rows).to_csv(a.out, mode="a", header=not os.path.exists(a.out), index=False)
        print(f"  wrote {len(rows)} rows -> {a.out}")

    if a.save_models_dir:
        import json, joblib
        tag = f"{a.model}_{a.classifier}_hetinfer_{base['mode']}_S{a.source.replace(':', '_')}" + \
              ("_delta" if args.delta else "")
        mdir = os.path.join(a.save_models_dir, a.benchmark, tag)
        os.makedirs(mdir, exist_ok=True)

        def _save(predictor, name):
            m = predictor.predictor.model
            (joblib.dump(m, os.path.join(mdir, name + ".joblib")) if a.model in ("dt", "svm")
             else m.save(os.path.join(mdir, name + ".keras")))

        _save(g_model[0], "global")
        for p in covered:
            _save(phase_models[p][0], f"phase{p}")
        with open(os.path.join(mdir, "metadata.json"), "w") as f:
            json.dump(dict(base, covered=covered, accepted=accepted,
                           mape={f"{r['regime']}/{r['method']}": r["mape"]
                                 for r in rows if r["phase"] == "all"}), f, indent=2)
        print(f"  saved ensemble + metadata -> {mdir}")


if __name__ == "__main__":
    main()

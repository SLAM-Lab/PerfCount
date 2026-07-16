#!/usr/bin/env python3
"""
predict_phase_forecast.py
=========================
Phase-level (per-phase) value forecasting, after Alcorta et al.

Question: does conditioning the next-interval counter forecaster on the current
program *phase* beat a single global forecaster?

Pipeline (reuses the repo's classify + forecasting stacks; they only met at the
data loader before this):
  1. get_raw_data -> per-benchmark counter trace (one row = one 10M-instr window).
  2. Detect phases with the existing classify() stack (gmm/2kmeans/pcakmeans/table)
     -> a per-window integer phase-id Series aligned to the trace index.
  3. Time-ordered split + supervised windows via get_separate_time_series_splits.
  4. Forecast the target counter (input_counters[0], e.g. ref_cycles) with four
     methods, all scored by MAPE against the SAME test targets:
       - global      : one forecaster on all training windows (the floor to beat).
       - per_phase   : a separate forecaster per phase; each test window is routed
                       to its phase's model (phases too small to train fall back to
                       the global model).
       - conditioned : one global forecaster with the phase id appended as an input
                       feature ("the model knows the current phase").
       - persistence : carry the last value forward (reference baseline).
  5. Report overall MAPE per method + a per-phase breakdown (n, target CoV, global
     vs per-phase MAPE) so we can see which phases benefit.

Note: phases are currently detected on the full trace (the unsupervised clusterer
sees the test region). That is fine for validating the mechanism; a leakage-clean
variant would fit the clusterer on train and predict phases on test.

Usage (single config, one benchmark):
  python predict_phase_forecast.py --benchmark aligned_dacapo_avrora_4.0GHz_cpu0 \
      --dataset x86_desktop_heterogeneous \
      --input_counters ref_cycles cpu_cycles branch_misses instructions \
      --model dt --classifier gmm --phase_count 6 --timesteps 5 --out results.csv
"""
import os
import sys
import argparse

import numpy as np
import pandas as pd

# Make the shared `src` package importable regardless of CWD.
WF_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WF_DIR not in sys.path:
    sys.path.insert(0, WF_DIR)

from src.input_parser import set_experiment_args, set_preprocess_args, supervised_model_args
from src.create_dataset import (
    get_raw_data,
    get_separate_time_series_splits,
    reshape_with_batch_size,
)
from src.classify import pre_classification, classify
import src.Predictor as pred
import src.evaluate as ev
from sklearn.tree import DecisionTreeClassifier


# ---------------------------------------------------------------------------
# args
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # Reuse the repo's standard arg groups (input_counters, model, scaler, etc.).
    set_experiment_args(p)
    set_preprocess_args(p)
    supervised_model_args(p)
    # Forecasting args (added by get_input_args only in the 'forecasting' branch).
    p.add_argument("--timesteps", type=int, default=5, help="input timesteps")
    p.add_argument("--forecast_horizon", type=int, default=1, help="output timesteps")
    # Phase-classification args (mirror input_parser.py:88-95).
    p.add_argument("--phase_count", type=int, default=6)
    p.add_argument("--classifier", default="gmm",
                   choices=["table", "2kmeans", "pcakmeans", "gmm"])
    p.add_argument("--classifier_threshold", type=float, default=1.0)
    p.add_argument("--distance_metric", default="euclidean",
                   choices=["euclidean", "manhattan"])
    p.add_argument("--W", type=int, default=100)
    p.add_argument("--N1", type=int, default=10)
    p.add_argument("--multicore_phases", default=None)
    # Harness-specific.
    p.add_argument("--conditioned", action="store_true",
                   help="also evaluate the phase-conditioned (phase-as-feature) model")
    p.add_argument("--min_phase_train", type=int, default=64,
                   help="min training windows to build a per-phase model; smaller "
                        "phases fall back to the global model at test time")
    p.add_argument("--gate_folds", type=int, default=4,
                   help="k for the k-fold within-train validation that decides, per "
                        "phase, whether the per-phase model beats global (selection gate)")
    p.add_argument("--no_gate", action="store_true",
                   help="skip the selection gate + per_phase_gated (it retrains k*phases "
                        "models; expensive for NN)")
    p.add_argument("--phase_pred_depth", type=int, default=8,
                   help="max_depth of the next-phase DecisionTreeClassifier (Alcorta-style "
                        "phase prediction: route the value forecaster to the PREDICTED next phase)")
    p.add_argument("--delta", action="store_true",
                   help="forecast the residual over persistence: predict x[i+1]-x[i] and "
                        "reconstruct x[i]+Δ, making persistence the zero-baseline the model can "
                        "only improve on")
    p.add_argument("--save_models_dir", default=None,
                   help="if set, persist the trained ensemble (global + per-phase models + "
                        "next-phase classifier + metadata.json) under "
                        "<dir>/<benchmark>/<model>_<classifier>[_delta]/")
    p.add_argument("--out", default=None, help="append results as CSV rows to this path")
    args = p.parse_args(argv)
    # get_input_args normally coerces --pca; replicate for pcakmeans.
    if args.pca:
        args.pca = int(args.pca) if str(args.pca).isdigit() else float(args.pca)
    return args


# ---------------------------------------------------------------------------
# forecasting primitives (mirror forecasting.py main())
# ---------------------------------------------------------------------------

def train_model(args, X_tr, y_tr, in_scaler=None, out_scaler=None):
    """Fit a forecaster on (X_tr,y_tr) and return (predictor, in_scaler, out_scaler)
    so it can later be applied to ANY test windows (needed for cross-phase routing by
    the next-phase predictor). Scalers are reused when given, else fit on the train."""
    train_pi = pred.PredictorInputs(args, X_tr, y_tr, in_scaler, out_scaler)
    predictor = pred.SerialPredictor(args, train_pi.X)
    m = predictor.predictor.model
    if args.model in ("dt", "svm"):
        m.fit(train_pi.X, train_pi.y)
    else:
        m.fit(train_pi.X, train_pi.y, batch_size=args.batch_size,
              epochs=args.epochs, shuffle=True, verbose=0)
    return predictor, train_pi.in_scaler, train_pi.out_scaler


def predict_model(args, predictor, X_te, y_te, in_scaler, out_scaler, invert_ts):
    """Predict X_te with an already-trained predictor; returns an inverse-scaled
    prediction DataFrame indexed like X_te."""
    if len(X_te) == 0:
        return pd.DataFrame(columns=y_te.columns)
    test_pi = pred.PredictorInputs(args, X_te, y_te, in_scaler, out_scaler, batch_padding=True)
    m = predictor.predictor.model
    if args.model in ("dt", "svm"):
        yhat = m.predict(test_pi.X)
        if args.forecast_horizon == 1:
            yhat = np.asarray(yhat).reshape(-1, 1)
    else:
        yhat = m.predict(test_pi.X, batch_size=args.batch_size, verbose=0)
    test_pi.add_predictions(args, yhat)
    return invert_ts.invert_predicted_transforms(test_pi.y_hat)


def fit_predict(args, X_tr, y_tr, X_te, y_te, in_scaler, out_scaler, invert_ts):
    """Convenience: train then predict (used by the selection gate)."""
    predictor, isc, osc = train_model(args, X_tr, y_tr, in_scaler, out_scaler)
    pred_df = predict_model(args, predictor, X_te, y_te, isc, osc, invert_ts)
    return pred_df, isc, osc


def score(val_all, pred_df):
    """Overall MAPE and per-window absolute-percentage-error series for pred_df,
    against the precomputed validation set val_all (true future per index)."""
    both = pd.concat([val_all.loc[pred_df.index], pred_df],
                     axis=1, keys=["True", "Prediction"]).dropna()
    true = both["True"].values
    pred = both["Prediction"].values
    mape = ev.get_regression_metrics(true, pred, 1)["mape"] if len(both) else float("nan")
    # per-window APE (mean over horizon columns), indexed like `both`.
    with np.errstate(divide="ignore", invalid="ignore"):
        ape = np.abs(100.0 * (true - pred) / true)
    ape = np.where(np.isfinite(ape), ape, np.nan).mean(axis=1)
    return mape, pd.Series(ape, index=both.index)


def gate_phases(args, X_tr, y_tr, ph_tr, covered, invert_ts):
    """Decide, per candidate phase, whether its dedicated model beats the global
    model -- using k-fold cross-validation *within the TRAIN region* (no test
    peeking). For every fold, an inner global model and inner per-phase models are
    fit on the other folds and scored on this fold's windows; each phase's per-window
    APEs are POOLED across folds (both per-phase and global, on the same windows).
    A phase is accepted iff its pooled per-phase APE beats pooled global APE. Pooling
    across folds makes the decision robust to any single unrepresentative slice. The
    final test predictions still come from the full-train per-phase models
    (standard select-then-use-full-model)."""
    if not covered:
        return []
    n = len(X_tr)
    k = max(2, args.gate_folds)
    if n < k * max(args.min_phase_train, 1):
        k = 2   # too little train to fold finely
    if n < 2 * args.min_phase_train:
        return covered   # can't validate; keep all (ungated behaviour)

    from sklearn.model_selection import KFold
    kf = KFold(n_splits=k, shuffle=False)   # contiguous folds respect time order

    g_pool = {p: [] for p in covered}   # pooled global APE on phase p's val windows
    p_pool = {p: [] for p in covered}   # pooled per-phase APE on phase p's val windows
    pos = np.arange(n)
    for inner_pos, val_pos in kf.split(pos):
        Xi, yi = X_tr.iloc[inner_pos], y_tr.iloc[inner_pos]
        Xv, yv = X_tr.iloc[val_pos], y_tr.iloc[val_pos]
        ph_i = ph_tr.reindex(Xi.index)
        ph_v = ph_tr.reindex(Xv.index)
        try:
            gi_pred, _, _ = fit_predict(args, Xi, yi, Xv, yv, None, None, invert_ts)
        except Exception:
            continue
        g_val = direct_ape(yv, gi_pred)
        ph_of_gval = ph_tr.reindex(g_val.index)
        for p in covered:
            imask = ph_i == p
            vmask = ph_v == p
            if imask.sum() < args.min_phase_train or vmask.sum() < 1:
                continue
            try:
                pi_pred, _, _ = fit_predict(args, Xi[imask.values], yi[imask.values],
                                            Xv[vmask.values], yv[vmask.values],
                                            None, None, invert_ts)
            except Exception:
                continue
            p_pool[p].append(direct_ape(yv[vmask.values], pi_pred).values)
            g_pool[p].append(g_val[ph_of_gval == p].values)

    accepted = []
    for p in covered:
        if not p_pool[p] or not g_pool[p]:
            continue
        pmean = np.concatenate(p_pool[p]).mean()
        gmean = np.concatenate(g_pool[p]).mean()
        if np.isfinite(pmean) and np.isfinite(gmean) and pmean < gmean:
            accepted.append(p)
    return accepted


def direct_ape(y_df, pred_df):
    """Per-window APE (original units) of pred_df against the supervised target y_df
    (its first column is the next-step true value). Used for the selection gate,
    where the true future is just y_df and no get_validation_set lookup is needed."""
    true = y_df.iloc[:, 0]
    idx = pred_df.index.intersection(true.index)
    t = true.loc[idx].values
    p = pred_df.loc[idx].iloc[:, 0].values if pred_df.ndim > 1 else pred_df.loc[idx].values
    with np.errstate(divide="ignore", invalid="ignore"):
        ape = np.abs(100.0 * (t - p) / t)
    s = pd.Series(ape, index=idx)
    return s[np.isfinite(s)]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv=None):
    args = parse_args(argv)
    target = args.input_counters[0]

    # 1. load trace ---------------------------------------------------------
    data = get_raw_data(args)
    if data is None or len(data) == 0:
        print(f"[skip] no data for {args.benchmark}")
        return
    data = data.reset_index(drop=True)

    # 2. detect phases on the trace ----------------------------------------
    scaled = pre_classification(data, args.filter_size, multicore=False)
    phases = classify(args, scaled).astype(int)
    phases.index = data.index
    n_phase = phases.nunique()

    # per-phase target CoV (stability); computed on the full trace.
    cov_by_phase = {}
    for p in sorted(phases.unique()):
        v = data.loc[phases == p, target]
        cov_by_phase[p] = float(v.std() / v.mean()) if v.mean() else float("nan")

    # 3. split + supervised windows ----------------------------------------
    X_tr, y_tr, X_te, y_te, train_ts, test_ts = get_separate_time_series_splits(args, data)
    X_tr, y_tr, X_te, y_te = reshape_with_batch_size(X_tr, y_tr, X_te, y_te, args.batch_size)
    if len(X_tr) == 0 or len(X_te) == 0:
        print(f"[skip] empty split for {args.benchmark}")
        return

    ph_tr = phases.reindex(X_tr.index)
    ph_te = phases.reindex(X_te.index)

    # next-phase labels + transition mask. A "transition" test window is one whose
    # phase differs from the next window's -- exactly where persistence mispredicts,
    # and where a forecaster must earn its keep. Used both for per_phase_pred routing
    # and for the transition-weighted metric.
    next_phase = phases.shift(-1)
    npt_tr = next_phase.reindex(X_tr.index)
    npt_te = next_phase.reindex(X_te.index)
    is_trans = (ph_te != npt_te) & npt_te.notna()   # bool Series on X_te.index

    # delta/residual targets (--delta): predict x[i+1]-x[i], reconstruct x[i]+Δ, so
    # persistence is the zero-baseline. Anchor x[i] = the target's 0-lag feature.
    anchor_col = f"t_{target}" if f"t_{target}" in X_tr.columns else target
    a_te = X_te[anchor_col] if args.delta else None
    if args.delta:
        if anchor_col not in X_tr.columns:
            print(f"[skip] --delta needs the target's 0-lag feature '{anchor_col}'")
            return
        y_tr = y_tr.sub(X_tr[anchor_col].values, axis=0)
        y_te = y_te.sub(X_te[anchor_col].values, axis=0)

    def reconstruct(pred_df):
        return pred_df if not args.delta else pred_df.add(a_te.reindex(pred_df.index), axis=0)

    # ground-truth future per test index (built once, method-agnostic).
    val_all = ev.get_validation_set("original", test_ts, target, 1,
                                    args.forecast_horizon).loc[X_te.index]

    # heterogeneous-history regime (training-injection): homogeneous (no injection),
    # naive (raw donor swap), or translated (donor unified to target config via CatBoost).
    het_prob = getattr(args, "heterogeneous_prob", 0.0)
    het_translated = bool(getattr(args, "cbm_model_dir", None) or
                          getattr(args, "cbm_cross_proc_dir", None))
    regime = "homogeneous" if het_prob <= 0 else ("translated" if het_translated else "naive")

    rows = []
    base = dict(bench=args.benchmark, dataset=args.dataset, target=target,
                model=args.model, classifier=args.classifier,
                phase_count=n_phase, timesteps=args.timesteps,
                horizon=args.forecast_horizon, regime=regime,
                het_prob=het_prob, het_mode=getattr(args, "heterogeneous_mode", "none"))

    def add(method, phase, mape, n, cov=np.nan):
        rows.append(dict(base, method=method, phase=phase, mape=mape, n=n, cov=cov))

    # 4a. global ------------------------------------------------------------
    g_predictor, g_in, g_out = train_model(args, X_tr, y_tr)
    g_pred = reconstruct(predict_model(args, g_predictor, X_te, y_te, g_in, g_out, test_ts))
    g_mape, g_ape = score(val_all, g_pred)
    add("global", "all", g_mape, len(g_ape))
    for p in sorted(ph_te.dropna().unique()):
        m = g_ape[ph_te.reindex(g_ape.index) == p]
        if len(m):
            add("global", int(p), float(m.mean()), len(m), cov_by_phase.get(int(p), np.nan))
    gtm = is_trans.reindex(g_ape.index).fillna(False).values
    if gtm.sum():
        add("global", "transition", float(g_ape[gtm].mean()), int(gtm.sum()))
    if (~gtm).sum():
        add("global", "steady", float(g_ape[~gtm].mean()), int((~gtm).sum()))

    # 4b. per-phase MODELS: train one per phase (shared global scaling), then predict
    # the FULL test set with each -- so any routing scheme can select from them.
    phase_models = {}
    for p in sorted(ph_tr.dropna().unique()):
        if (ph_tr == p).sum() < args.min_phase_train:
            continue   # too few train windows -> that phase falls back to global
        Xtr_p, ytr_p = X_tr[(ph_tr == p).values], y_tr[(ph_tr == p).values]
        try:
            phase_models[int(p)] = train_model(args, Xtr_p, ytr_p, g_in, g_out)
        except Exception as e:
            print(f"[warn] per-phase model p={p} failed ({type(e).__name__}: {e})")
    covered = sorted(phase_models)
    phase_pred_full = {
        p: reconstruct(predict_model(args, phase_models[p][0], X_te, y_te,
                                     phase_models[p][1], phase_models[p][2], test_ts))
        for p in covered
    }

    def route(route_series, allowed=None):
        """Full test prediction: window i uses phase_models[route(i)] where that phase
        is allowed and covered, else the global model."""
        allowed = covered if allowed is None else allowed
        pred = g_pred.copy()
        rs = route_series.reindex(pred.index)
        for p in allowed:
            idx = rs.index[rs == p]
            if len(idx):
                pred.loc[idx] = phase_pred_full[p].loc[idx]
        return pred.sort_index()

    def report(method, pred):
        mape, ape = score(val_all, pred)
        add(method, "all", mape, len(ape))
        for p in sorted(ph_te.dropna().unique()):
            m = ape[ph_te.reindex(ape.index) == p]
            if len(m):
                add(method, int(p), float(m.mean()), len(m), cov_by_phase.get(int(p), np.nan))
        # transition-weighted split: how the method does at phase transitions (where
        # persistence is worst) vs steady windows.
        tm = is_trans.reindex(ape.index).fillna(False).values
        if tm.sum():
            add(method, "transition", float(ape[tm].mean()), int(tm.sum()))
        if (~tm).sum():
            add(method, "steady", float(ape[~tm].mean()), int((~tm).sum()))
        return mape

    all_train_phases = sorted(int(x) for x in ph_tr.dropna().unique())
    print(f"  per-phase models for phases {covered} (of {all_train_phases})")

    # per_phase: route on the CURRENT window's phase (assumes phase persists 1 step).
    pp_mape = report("per_phase", route(ph_te))

    # 4b'. per_phase_pred: Alcorta-style -- PREDICT the next phase, route to it.
    # Learned next-phase classifier on the same lagged features (npt_* computed above).
    trm = npt_tr.notna()
    clf = DecisionTreeClassifier(max_depth=args.phase_pred_depth, random_state=42)
    clf.fit(X_tr[trm.values].values, npt_tr[trm].astype(int).values)
    pred_next = pd.Series(clf.predict(X_te.values), index=X_te.index)
    tem = npt_te.notna()
    trans_rate = float((npt_te[tem].astype(int).values != ph_te[tem].astype(int).values).mean()) \
        if tem.sum() else float("nan")
    pred_acc = float((pred_next[tem].astype(int).values == npt_te[tem].astype(int).values).mean()) \
        if tem.sum() else float("nan")
    ppp_mape = report("per_phase_pred", route(pred_next))
    # oracle upper bound: route on the TRUE next phase (how much a perfect phase
    # predictor could buy).
    oracle_mape = report("per_phase_oracle_next", route(npt_te.fillna(-1).astype(int)))
    print(f"  next-phase: transition_rate={trans_rate:.3f}  predictor_acc={pred_acc:.3f}")
    add("phase_pred_stats", "all", pred_acc, tem.sum(), trans_rate)

    # 4b''. selection gate (skip with --no_gate; expensive for NN).
    gated_mape = None
    accepted = None
    if not args.no_gate:
        accepted = gate_phases(args, X_tr, y_tr, ph_tr, covered, test_ts)
        gated_mape = report("per_phase_gated", route(ph_te, accepted))
        print(f"  gate accepted phases {accepted} (of {covered})")

    # 4c. phase-conditioned (phase id as an input feature) ------------------
    if args.conditioned and not args.delta:   # conditioned path is absolute-only
        data_c = data.copy()
        data_c["phase_id"] = phases.reindex(data_c.index).astype(float)
        Xc_tr, yc_tr, Xc_te, yc_te, _, test_ts_c = get_separate_time_series_splits(args, data_c)
        Xc_tr, yc_tr, Xc_te, yc_te = reshape_with_batch_size(
            Xc_tr, yc_tr, Xc_te, yc_te, args.batch_size)
        c_pred, _, _ = fit_predict(args, Xc_tr, yc_tr, Xc_te, yc_te, None, None, test_ts_c)
        val_all_c = ev.get_validation_set("original", test_ts_c, target, 1,
                                          args.forecast_horizon).loc[Xc_te.index]
        c_mape, c_ape = score(val_all_c, c_pred)
        add("conditioned", "all", c_mape, len(c_ape))

    # 4d. persistence (with transition/steady split) ------------------------
    orig = test_ts.df["original"][target]
    true_next = orig.shift(-args.forecast_horizon)
    persist_idx = X_te.index.intersection(orig.index)
    tp = true_next.loc[persist_idx].values
    pp_ = orig.loc[persist_idx].values
    with np.errstate(divide="ignore", invalid="ignore"):
        ape_pw = pd.Series(np.abs(100.0 * (tp - pp_) / tp), index=persist_idx)
    ape_pw = ape_pw[np.isfinite(ape_pw)]
    persist_mape = float(ape_pw.mean()) if len(ape_pw) else float("nan")
    add("persistence", "all", persist_mape, len(ape_pw))
    tmp = is_trans.reindex(ape_pw.index).fillna(False).values
    if tmp.sum():
        add("persistence", "transition", float(ape_pw[tmp].mean()), int(tmp.sum()))
    if (~tmp).sum():
        add("persistence", "steady", float(ape_pw[~tmp].mean()), int((~tmp).sum()))

    # 5. report + persist ---------------------------------------------------
    mode = "delta" if args.delta else "absolute"
    print(f"\n{args.benchmark}  [{args.model} / {args.classifier} / "
          f"{n_phase} phases / {mode}]  target={target}")

    def trans_of(method):
        r = [x for x in rows if x["method"] == method and x["phase"] == "transition"]
        return r[0]["mape"] if r else float("nan")

    print(f"  {'method':22s} {'MAPE':>6}  {'@transition':>11}  {'@steady':>8}")
    def line(name, mape):
        st = [x for x in rows if x["method"] == name and x["phase"] == "steady"]
        print(f"  {name:22s} {mape:6.2f}  {trans_of(name):11.2f}  {(st[0]['mape'] if st else float('nan')):8.2f}")
    line("global", g_mape)
    line("per_phase", pp_mape)
    line("per_phase_pred", ppp_mape)
    line("per_phase_oracle_next", oracle_mape)
    if gated_mape is not None:
        line("per_phase_gated", gated_mape)
    if args.conditioned and not args.delta:
        line("conditioned", c_mape)
    line("persistence", persist_mape)

    out = pd.DataFrame(rows)
    if args.out:
        header = not os.path.exists(args.out)
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        out.to_csv(args.out, mode="a", header=header, index=False)
        print(f"  wrote {len(out)} rows -> {args.out}")

    # 6. persist the trained ensemble + metadata (reproducibility / deployment).
    if args.save_models_dir:
        import json
        import joblib
        tag = f"{args.model}_{args.classifier}" + ("_delta" if args.delta else "")
        mdir = os.path.join(args.save_models_dir, args.benchmark, tag)
        os.makedirs(mdir, exist_ok=True)

        def _save_predictor(predictor, name):
            m = predictor.predictor.model
            if args.model in ("dt", "svm"):
                joblib.dump(m, os.path.join(mdir, name + ".joblib"))
            else:
                m.save(os.path.join(mdir, name + ".keras"))

        _save_predictor(g_predictor, "global")
        for p in covered:
            _save_predictor(phase_models[p][0], f"phase{p}")
        joblib.dump(clf, os.path.join(mdir, "next_phase_clf.joblib"))

        def _by_phase(phase_key):
            return {r["method"]: r["mape"] for r in rows if r["phase"] == phase_key}

        meta = {
            "benchmark": args.benchmark, "dataset": args.dataset, "model": args.model,
            "classifier": args.classifier, "phase_count": int(n_phase), "mode": mode,
            "target": target, "timesteps": args.timesteps, "horizon": args.forecast_horizon,
            "epochs": args.epochs, "covered_phases": covered, "accepted_phases": accepted,
            "transition_rate": trans_rate, "next_phase_acc": pred_acc,
            "mape": _by_phase("all"), "mape_transition": _by_phase("transition"),
            "mape_steady": _by_phase("steady"),
        }
        with open(os.path.join(mdir, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  saved ensemble + metadata -> {mdir}")


if __name__ == "__main__":
    main()

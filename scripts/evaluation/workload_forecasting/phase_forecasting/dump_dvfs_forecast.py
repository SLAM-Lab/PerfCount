#!/usr/bin/env python3
"""dump_dvfs_forecast.py
========================
Walk-forward DVFS workload-forecasting dump for the scheduling simulator.

For one (core, source-frequency) run of a workload, forecast every OTHER same-core
target frequency's ref_cycles per 10M-instruction chunk, causally, and serialize the
result in the exact `cross_freq_precompute.py` layout so the simulator's Model_Forecast
DVFS policy can consume it.

Method (self-supervised on the translated-source stream -- the honest cold-prediction
setup): at runtime the workload is on the source config S; we observe S's counters,
translate them to each target config C (ref_cycles via the cross-freq CatBoost tree;
config-invariant counters copied), and NEVER observe C. The forecaster is trained
self-supervised on the translated-S trace via EXPANDING-WINDOW walk-forward -- warm up
on the first chunks, then predict forward in blocks, retraining on all history-so-far
each block. Only the past predicts the future (no leakage); coverage is the whole trace
minus a short warmup (filled with translated-persistence, the baseline floor).

Output: one CSV per (source-freq, phase) in
    <out_dir>/speedups_from_{P|E}_<src>GHz/speedups_{P|E}_<src>GHz_<bench>_phase<ph>.csv
with columns  sample_index, Time_{P|E}_<src>GHz, Speedup_{P|E}_<tgt>GHz_vs_{P|E}_<src>GHz (x3)
matching data_loader.load_phase_data. The loader recovers tgt_time = Time_src / Speedup,
so Time_src is just a carrier and cancels; we set Speedup = Time_src / forecast_tgt_time.

Usage:
  dump_dvfs_forecast.py --bench spec_505.mcf_r --core 0 --source_freq 4.0 \
      --out_dir results/scheduling/forecast_predictions --model dt
"""
import argparse, glob, os, re, sys
import numpy as np, pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_WF = os.path.dirname(_HERE)                              # workload_forecasting/
_CC = os.path.join(_WF, "heterogeneous_forecasting", "inference")   # predict_cross_config
if not os.path.isdir(_CC):
    raise RuntimeError(f"cross-config inference dir not found: {_CC}")
for p in (_WF, _CC, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
from src.create_dataset import get_transformed_time_series, BENCHMARK_NAME_NOPHASE_RE  # noqa
from src.classify import pre_classification, classify                                   # noqa
import predict_phase_forecast as pf                                                     # noqa
import predict_cross_config as cc                                                       # noqa
from types import SimpleNamespace

TARGET = "ref_cycles"
FREQS = [1.0, 2.0, 3.0, 4.0]
PREFIX = {"0": "P", "16": "E"}
DATA_DIR = cc.DATA_DIR


def build_args(counters, model, timesteps, classifier, phase_count, epochs, filter_size,
               gate_window=200, gate_margin=0.05):
    return SimpleNamespace(
        input_counters=list(counters), model=model, timesteps=timesteps,
        forecast_horizon=1, scaler="minmax", filter="none", filter_size=filter_size,
        pca=None, train_size=70, batch_size=32, start_drop_count=0, end_drop_count=0,
        heterogeneous_prob=0.0, add_heterogeneity_features=False,
        classifier=classifier, phase_count=phase_count, classifier_threshold=1.0,
        distance_metric="euclidean", W=100, N1=10, multicore_phases=None,
        epochs=epochs, neurons=16, stateless=True, dense_hidden_layers=[50, 50],
        early_stopping=False, patience=5, loss_function="mse", regression_activation="linear",
        optimizer="adam", stacked_layers=2, tree_max_depth=3, num_heads=2, dropout_rate=0.2,
        svm_kernel="rbf", svm_regularization=1.0, svm_epsilon=0.1, max_iter=-1,
        min_phase_train=64, gate_folds=4, no_gate=True, delta=True, phase_pred_depth=8,
        gate_window=gate_window, gate_margin=gate_margin,
    )


def source_phase_lengths(bench, freq, cpu):
    """[(phase_num, nrows), ...] in concat order = sorted phase number (matches load_full_trace)."""
    wl = cc.workload_name(bench, f"{freq}", f"{cpu}")
    single = glob.glob(os.path.join(DATA_DIR, "**", f"{wl}.csv"), recursive=True)
    if single:
        return [(0, len(pd.read_csv(single[0], usecols=[0])))]
    paths = glob.glob(os.path.join(DATA_DIR, "**", f"{wl}_phase*.csv"), recursive=True)
    by_num = {}
    for p in sorted(paths):
        n = int(re.search(r"_phase(\d+)\.csv$", p).group(1))
        by_num.setdefault(n, p)
    return [(n, len(pd.read_csv(by_num[n], usecols=[0]))) for n in sorted(by_num)]


# Coverage of the per-phase models: how many predicted chunks actually got a
# phase-specific model vs silently fell back to the global one.
_PHASE_STATS = {'total': 0, 'phase_model': 0, 'skip_too_few': 0,
                'skip_train_failed': 0, 'global_only': 0}

# Gate decisions, so the gates can be reported rather than assumed to have fired.
_GATE_STATS = {'persist_checked': 0, 'persist_fired': 0, 'persist_chunks': 0,
               'phase_checked': 0, 'phase_accepted': 0, 'phase_rejected': 0,
               'phase_rejected_chunks': 0, 'phase_no_val': 0}


def walk_forward(args, frame, method, warmup, block, max_train, horizon=1, gate='none'):
    """Expanding-window causal forecast of `frame`'s ref_cycles. With horizon=1
    predicts the next chunk; with horizon=K predicts the MEAN of the next K chunks
    (the upcoming window) -- a multi-step forecast whose advantage over persistence
    grows with K, and which lets a policy commit to the config best for the window.

    Returns a Series keyed by the chunk the value is a forecast OF. The model at label i
    consumes features from chunk i and predicts chunk i+1 (verified: anchor[i]==raw[i],
    y[i]==raw[i+1]), so its output is filed at i+1. That matches the caller's fallback
    array, whose base[i]=raw[i-1] is a persistence forecast OF chunk i, and it matches
    the simulator, which reads row i as the causal forecast of chunk i. Filing the model
    at label i instead would put a forecast of chunk i+1 in chunk i's row."""
    ts = get_transformed_time_series(args, frame[args.input_counters])
    Xall = ts.X
    anchor_col = f"t_{TARGET}" if f"t_{TARGET}" in Xall.columns else TARGET
    anchor = Xall[anchor_col]
    if horizon > 1:
        # forward K-window mean of the target: fwd[i] = mean(target[i+1 .. i+K])
        tser = frame[TARGET].astype(float)
        fwd = tser.rolling(horizon).mean().shift(-horizon)
        yall = fwd.reindex(Xall.index).to_frame(name=TARGET)
    else:
        yall = ts.y if isinstance(ts.y, pd.DataFrame) else ts.y.to_frame()
    ydelta = yall.sub(anchor.values, axis=0)                     # delta target (DataFrame)

    ph_full = classify(args, pre_classification(frame[args.input_counters], args.filter_size)).astype(int)
    ph_full.index = frame.index
    ph = ph_full.reindex(Xall.index)

    ix = Xall.index
    N = len(ix)
    W = min(max(warmup, args.min_phase_train), N - 1)
    out = {}
    for start in range(W, N, block):
        lo = max(0, start - max_train) if max_train > 0 else 0
        trl = ix[lo:start]
        tel = ix[start:min(start + block, N)]
        if len(tel) == 0:
            break
        yv = ydelta.loc[trl].dropna()                            # drop tail rows w/o full window
        if len(yv) < args.min_phase_train:
            continue

        # The PHASE gate needs an honest held-out slice to decide on. Take it from the TAIL
        # of the training window (the most recent history), which is causal and is also the
        # part most like the block we are about to predict.
        #
        # Only the phase gate uses val_idx. The persistence gate decides online from a
        # trailing window of realized error, so holding data out for it buys nothing and
        # costs the model 20% of its training window -- a handicap that showed up as the
        # gated arm's predictions diverging from the ungated arm's by ~0.6% on chunks where
        # the gate had chosen the model. Hold out only when a gate actually consumes it.
        fit_idx, val_idx = yv.index, None
        if gate in ('phase', 'both') and len(yv) >= 2 * args.min_phase_train:
            n_val = max(args.min_phase_train, int(0.2 * len(yv)))
            fit_idx, val_idx = yv.index[:-n_val], yv.index[-n_val:]

        g = pf.train_model(args, Xall.loc[fit_idx], yv.loc[fit_idx])

        def predict(model_tuple, labels):
            pdf = pf.predict_model(args, model_tuple[0], Xall.loc[labels],
                                   ydelta.loc[labels].fillna(0.0),   # target unused for predict
                                   model_tuple[1], model_tuple[2], ts)
            return pdf.add(anchor.loc[labels], axis=0)          # reconstruct absolute

        def _mape(pred_abs, labels):
            """MAPE of an absolute prediction against the true target on `labels`."""
            tv = yall.loc[labels]
            tv = tv.iloc[:, 0] if isinstance(tv, pd.DataFrame) else tv
            pv = pred_abs.iloc[:, 0] if isinstance(pred_abs, pd.DataFrame) else pred_abs
            m = np.isfinite(tv.values) & np.isfinite(pv.values) & (tv.values > 0)
            if m.sum() == 0:
                return np.inf
            return float(np.mean(np.abs(pv.values[m] - tv.values[m]) / tv.values[m]))

        # ---- GATE: persistence (online, per chunk) ----------------------------------
        # Forecasting ADDS error on workloads the previous chunk already predicts well.
        # Measured on the corrected self-forecast dumps (147 bench-phases x 8 configs), the
        # ungated model LOSES to persistence by 0.77pp on chunks where persistence is under
        # 2% MAPE, and WINS by 3.74pp only where persistence exceeds 10%.
        #
        # The gate is evaluated per chunk, not per block: chunk i-1's true value is known
        # before chunk i is scheduled (the scheduler just ran it), so a trailing-window
        # comparison of model-vs-persistence error is causal and adapts inside a block.
        # A block-level gate commits for 4096 chunks and was measurably too coarse.
        #
        # This bounds the forecaster below by persistence -- which is exactly the reactive
        # baseline in the simulator -- so a gated forecaster should never be much worse
        # than reactive, and can only win where forecasting genuinely helps.
        gate_persist = gate in ('persist', 'both')

        pred = predict(g, tel)
        if method != "global":
            # Per-phase models overwrite the global prediction for chunks whose phase has
            # enough training samples. Where they do not (too few samples, or a training
            # failure) those chunks silently keep the GLOBAL prediction -- i.e. this block
            # degrades toward phase-unaware without saying so. Count the coverage so the
            # "phase-aware" label can be checked rather than assumed.
            ph_tr = ph.loc[fit_idx]
            for pv in sorted(ph_tr.dropna().unique()):
                if (ph_tr == pv).sum() < args.min_phase_train:
                    _PHASE_STATS['skip_too_few'] += int((ph.loc[tel] == pv).sum())
                    continue
                sub = tel[ph.loc[tel] == pv]
                if len(sub) == 0:
                    continue
                try:
                    si = fit_idx[(ph_tr == pv).values]
                    pm = pf.train_model(args, Xall.loc[si], yv.loc[si], g[1], g[2])

                    # ---- GATE: phase model --------------------------------------------
                    # Splitting an already-capped training window by phase trades samples
                    # for specificity, and measured over 68 benchmarks it loses that trade
                    # on 63 of them (+0.85pp chunk-weighted; 100% of SPEC26). Only adopt
                    # the phase model where it actually beats the global model on this
                    # phase's held-out recent history.
                    if gate in ('phase', 'both') and val_idx is not None:
                        vsub = val_idx[(ph.loc[val_idx] == pv).values]
                        if len(vsub) < 3:
                            _GATE_STATS['phase_no_val'] += len(sub)
                            continue
                        _GATE_STATS['phase_checked'] += 1
                        if _mape(predict(pm, vsub), vsub) >= _mape(predict(g, vsub), vsub):
                            _GATE_STATS['phase_rejected'] += 1
                            _GATE_STATS['phase_rejected_chunks'] += len(sub)
                            continue
                        _GATE_STATS['phase_accepted'] += 1

                    pp = predict(pm, sub)
                    pred.loc[pp.index] = pp
                    _PHASE_STATS['phase_model'] += len(sub)
                except Exception:
                    _PHASE_STATS['skip_train_failed'] += len(sub)
        else:
            _PHASE_STATS['global_only'] += len(tel)
        _PHASE_STATS['total'] += len(tel)
        col = pred.columns[0] if isinstance(pred, pd.DataFrame) else None
        if not gate_persist:
            for lab in tel:
                out[lab + 1] = float(pred.loc[lab, col] if col is not None else pred.loc[lab])
        else:
            ytrue = yall.iloc[:, 0] if isinstance(yall, pd.DataFrame) else yall
            em, ep = [], []          # trailing realized errors: model, persistence
            for lab in tel:
                pm = float(pred.loc[lab, col] if col is not None else pred.loc[lab])
                # Persistence forecast OF chunk lab+1 is the value at chunk lab, i.e.
                # anchor[lab]. anchor[i]==raw[i] is the CURRENT chunk, so for the chunk
                # being predicted (lab+1) it is the previous one -- the correct baseline,
                # and exactly what the reactive policy reads. (An earlier version compared
                # against anchor at the predicted chunk, which is ground truth, making the
                # gate an oracle and its results meaningless.)
                pp = float(anchor.loc[lab])
                _GATE_STATS['persist_checked'] += 1
                # Decide using ONLY chunks already executed. Require the model to beat
                # persistence by `gate_margin` to switch away from the safe default.
                if len(em) >= args.gate_window // 4:
                    mm, mp = float(np.mean(em[-args.gate_window:])), float(np.mean(ep[-args.gate_window:]))
                    use_model = mm * (1.0 + args.gate_margin) < mp
                else:
                    use_model = True          # no evidence yet: fall through to the model
                out[lab + 1] = pm if use_model else pp
                if not use_model:
                    _GATE_STATS['persist_fired'] += 1
                    _GATE_STATS['persist_chunks'] += 1
                tv = float(ytrue.loc[lab]) if lab in ytrue.index else np.nan   # = raw[lab+1]
                if np.isfinite(tv) and tv > 0:
                    em.append(abs(pm - tv) / tv)
                    ep.append(abs(pp - tv) / tv)
    return pd.Series(out, name=TARGET)


def _report_gates():
    g = _GATE_STATS
    if g['persist_checked']:
        print(f"  persistence gate: fired on {g['persist_fired']}/{g['persist_checked']} blocks "
              f"({g['persist_fired'] / g['persist_checked'] * 100:.0f}%), "
              f"{g['persist_chunks']} chunks emitted as persistence "
              f"(model did not beat the previous chunk on held-out history)")
    if g['phase_checked'] or g['phase_no_val']:
        tot = g['phase_checked']
        acc = g['phase_accepted']
        print(f"  phase gate: accepted {acc}/{tot} phase models"
              + (f" ({acc / tot * 100:.0f}%)" if tot else "")
              + f", rejected {g['phase_rejected']} ({g['phase_rejected_chunks']} chunks kept the "
                f"global model), {g['phase_no_val']} chunks had too little validation data")


def _report_phase_coverage(method):
    t = _PHASE_STATS['total']
    if not t:
        return
    pm = _PHASE_STATS['phase_model']
    print(f"  phase-model coverage [{method}]: {pm}/{t} predicted chunks ({pm / t * 100:.1f}%) "
          f"used a phase-specific model; fell back to global on "
          f"{_PHASE_STATS['skip_too_few']} (too few samples) + "
          f"{_PHASE_STATS['skip_train_failed']} (train failed) + "
          f"{_PHASE_STATS['global_only']} (method=global)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", required=True, help="stem, e.g. spec_505.mcf_r")
    ap.add_argument("--core", required=True, choices=["0", "16"], help="source core; 0=P, 16=E")
    ap.add_argument("--target_core", default=None, choices=["0", "16"],
                    help="target core (default = source core). If different -> CROSS-PROC mode "
                         "(migration): predict all 4 target-core freqs, cross_proc output format.")
    ap.add_argument("--source_freq", required=True, type=float)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--counters", nargs="+", default=["ref_cycles", "cpu_cycles", "branches", "instructions"])
    ap.add_argument("--model", default="dt")
    ap.add_argument("--classifier", default="gmm")
    ap.add_argument("--phase_count", type=int, default=6)
    ap.add_argument("--timesteps", type=int, default=5)
    ap.add_argument("--filter_size", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--method", default="per_phase", choices=["global", "per_phase"],
                    help="per_phase = train per-runtime-phase models on top of a global one "
                         "(phase-AWARE); global = one model, no phase segmentation (phase-UNAWARE).")
    ap.add_argument("--gate", default="none", choices=["none", "phase", "persist", "both"],
                    help="Adopt a specialized predictor only where it beats the simpler one on "
                         "held-out recent history. 'phase': use a per-phase model only if it "
                         "beats the global model on that phase (measured: unconditional "
                         "per-phase loses on 63/68 benchmarks). 'persist': emit persistence "
                         "unless the model beats it (measured: forecasting ADDS error where "
                         "persistence is already under ~5 pct MAPE; 736.ocio was 2.9 pct persistence "
                         "vs 54.7 pct forecast). 'both' applies both gates.")
    ap.add_argument("--mode", default="forecast", choices=["forecast", "translate", "self_forecast"],
                    help="forecast = walk-forward causal forecast (Model_Forecast); "
                         "translate = per-chunk ACTUAL-counter translation (reactive-model baseline).")
    ap.add_argument("--gate_window", type=int, default=200,
                    help="Trailing chunks of realized error used by the persistence gate.")
    ap.add_argument("--gate_margin", type=float, default=0.05,
                    help="Relative margin the model must beat persistence by before the gate "
                         "lets it through. >0 biases toward persistence (the safe default), "
                         "which is what bounds a gated forecaster below by the reactive policy.")
    ap.add_argument("--warmup", type=int, default=2048)
    ap.add_argument("--block", type=int, default=4096, help="chunks predicted per retrain")
    ap.add_argument("--max_train", type=int, default=20000,
                    help="sliding training-window cap (0 = expanding/unbounded)")
    ap.add_argument("--horizon", type=int, default=1,
                    help="forecast the MEAN of the next K chunks (K>1 = multi-step window forecast)")
    a = ap.parse_args()

    cpu, sf = a.core, a.source_freq
    t_cpu = a.target_core or cpu
    is_xp = (t_cpu != cpu)                                      # cross-proc (migration) mode
    prefix, t_prefix = PREFIX[cpu], PREFIX[t_cpu]
    src_ghz = f"{sf:.1f}GHz"
    args = build_args(a.counters, a.model, a.timesteps, a.classifier, a.phase_count, a.epochs,
                      a.filter_size, gate_window=a.gate_window, gate_margin=a.gate_margin)

    fullS = cc.load_full_trace(cc.workload_name(a.bench, f"{sf}", cpu))
    if fullS is None or not all(c in fullS.columns for c in a.counters):
        print(f"[skip] missing source trace/counters for {a.bench} cpu{cpu}@{src_ghz}")
        return
    fullS = fullS.reset_index(drop=True)
    n0 = len(fullS)
    time_src = fullS[TARGET].values / 1e9                       # source-config time carrier

    # cross-proc: predict ALL target-core freqs (incl. iso-freq migration). same-core: other freqs.
    # self_forecast: the source config ITSELF -- translation is identity, so the only
    # error left is forecast error. This is the Forecast x Oracle cell: it isolates how
    # good the forecaster is when the cross-platform model is perfect, which no other
    # policy in the grid can measure (every other forecast number folds translation
    # error in with forecast error).
    if a.mode == "self_forecast":
        if is_xp:
            print("[skip] --mode self_forecast is same-core only (target_core must equal core)")
            return
        tgt_freqs = [sf]
    else:
        tgt_freqs = list(FREQS) if is_xp else [f for f in FREQS if f != sf]
    tgt_time, tgt_base = {}, {}
    for tf in tgt_freqs:
        # translate S -> C: ref_cycles (+ cpu_cycles cross-core) via the cross-freq/cross-proc trees;
        # config-invariant counters (branches, instructions) copied.
        fS = fullS[a.counters].copy()
        tr = cc.translate_counter(fullS, a.bench, TARGET, cpu, f"{sf}", t_cpu, f"{tf}")
        if tr is None:
            print(f"[warn] no translator {cpu}:{sf}->{t_cpu}:{tf} for {a.bench}; skipping target")
            continue
        fS[TARGET] = tr
        if is_xp and "cpu_cycles" in a.counters:
            trc = cc.translate_counter(fullS, a.bench, "cpu_cycles", cpu, f"{sf}", t_cpu, f"{tf}")
            if trc is not None:
                fS["cpu_cycles"] = trc
        base = np.empty(n0); base[1:] = fS[TARGET].values[:-1]; base[0] = fS[TARGET].values[0]
        tgt_base[tf] = base / 1e9
        if a.mode == "translate":
            # reactive-model baseline: per-chunk ACTUAL-counter translation (no forecast).
            tgt_time[tf] = fS[TARGET].values.astype(float) / 1e9
        else:
            tt = (base / 1e9).copy()                            # translated-persistence fallback
            pred = walk_forward(args, fS, a.method, a.warmup, a.block, a.max_train, a.horizon,
                                gate=a.gate)
            for lab, v in pred.items():
                if 0 <= lab < n0:
                    tt[lab] = v / 1e9
            tgt_time[tf] = tt

    if not tgt_time:
        print(f"[skip] no targets for {a.bench} cpu{cpu}@{src_ghz} -> cpu{t_cpu}")
        return

    if a.mode == "self_forecast":
        # One file per config holding the causal forecast of that config's own true time.
        # No source axis: the forecast of config C depends only on C's history, not on
        # where the workload happened to run. The loader broadcasts it across sources.
        out_sub = os.path.join(a.out_dir, f"{prefix}_{src_ghz}")
        os.makedirs(out_sub, exist_ok=True)
        r0 = 0
        for ph, seglen in source_phase_lengths(a.bench, sf, cpu):
            r1 = r0 + seglen
            sl = slice(r0, r1)
            pd.DataFrame({
                "sample_index": np.arange(seglen),
                f"Time_pred_{prefix}_{src_ghz}": tgt_time[sf][sl],
                f"Time_true_{prefix}_{src_ghz}": time_src[sl],
            }).to_csv(os.path.join(out_sub, f"{a.bench}_phase{ph}.csv"), index=False)
            r0 = r1
        print(f"  wrote {len(source_phase_lengths(a.bench, sf, cpu))} phase file(s) -> {out_sub}/  "
              f"[self_forecast cpu{cpu}@{sf}GHz, n={n0}]")
        _report_phase_coverage(a.method)
        _report_gates()
        return

    out_sub = os.path.join(a.out_dir, f"speedups_from_{prefix}_{src_ghz}")
    os.makedirs(out_sub, exist_ok=True)
    r0 = 0
    for ph, seglen in source_phase_lengths(a.bench, sf, cpu):
        r1 = r0 + seglen
        sl = slice(r0, r1)
        ts = time_src[sl]
        cols = {"sample_index": np.arange(seglen), f"Time_{prefix}_{src_ghz}": ts}
        for tf in tgt_freqs:
            if tf not in tgt_time:
                continue
            spd = ts / np.where(tgt_time[tf][sl] > 0, tgt_time[tf][sl], np.nan)
            if is_xp:
                # cross-proc speedup has no frequency-ratio physics; guard forecast blowups by
                # keeping within 2x of the reactive-translation (persistence) speedup.
                ref = ts / np.where(tgt_base[tf][sl] > 0, tgt_base[tf][sl], np.nan)
                cols[f"Speedup_{t_prefix}_{tf:.1f}GHz_vs_{prefix}_{src_ghz}"] = np.clip(spd, ref * 0.5, ref * 2.0)
            else:
                # same-core: speedup in [min(1,tf/sf), max(1,tf/sf)] +/-5% (tight; guards ED2P).
                lo, hi = min(1.0, tf / sf), max(1.0, tf / sf)
                cols[f"Speedup_{prefix}_{tf:.1f}GHz_vs_{prefix}_{src_ghz}"] = np.clip(spd, lo * 0.95, hi * 1.05)
        # cross-proc filename has NO 'speedups_' prefix (cross_proc_precompute layout); same-core does.
        fname = (f"{a.bench}_phase{ph}.csv" if is_xp
                 else f"speedups_{prefix}_{src_ghz}_{a.bench}_phase{ph}.csv")
        pd.DataFrame(cols).to_csv(os.path.join(out_sub, fname), index=False)
        r0 = r1
    print(f"  wrote {len(source_phase_lengths(a.bench, sf, cpu))} phase file(s) -> {out_sub}/  "
          f"[{'cross_proc' if is_xp else 'cross_freq'} cpu{cpu}:{sf}->cpu{t_cpu}, "
          f"targets {sorted(tgt_time)}, n={n0}]")
    _report_phase_coverage(a.method)
    _report_gates()


if __name__ == "__main__":
    main()

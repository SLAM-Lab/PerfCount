#!/usr/bin/env python3
"""stitch_history.py -- forecast from the history the workload ACTUALLY had.

The existing dump (dump_dvfs_forecast.py) builds one universe per source configuration: it
takes that configuration's whole trace, translates every chunk to each target, and forecasts
from that stream. The simulator then reads the slice matching wherever the policy currently
sits. So the moment a policy migrates from A to B it starts reading a forecast conditioned on
B's history for every earlier chunk -- a history that never happened in that run.

This builds the patchwork instead. Given the configuration the policy actually occupied at
each chunk, the history of target C is assembled chunk by chunk:

    chunk j ran on C          ->  C's own measured counters       (no model)
    chunk j ran on some s!=C  ->  translate_counter(s -> C) at j  (model)

and the forecaster runs once over that stitched series.

Two consequences worth knowing.

The result does not depend on which source slice the simulator reads: "where am I now" is
already encoded in the stitched history. So the same forecast is written into every
speedups_from_* slice. Since the loader recovers tgt_time = Time_src / Speedup, each slice
keeps its own Time_src carrier and stores Speedup = Time_src / stitched_forecast, and every
slice decodes to the same target time. No loader change is needed.

It is also cheaper than the existing dump: one forecast per target (8) rather than one per
(source, target) pair (28), because the source axis has collapsed.

The path is policy- and metric-specific, so the output is too. Build one set per (policy,
metric) and read each metric's rows from its own simulator run.

Usage:
  stitch_history.py --bench spec_505.mcf_r --actions_dir results/scheduling/actions \
      --policy Model_Forecast_ReactiveGated_Hetero --metric EDP \
      --out_proc <dir> --out_freq <dir>
"""
import argparse, glob, os, re, sys
import numpy as np, pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_WF = os.path.dirname(_HERE)
_CC = os.path.join(_WF, "heterogeneous_forecasting", "inference")
if not os.path.isdir(_CC):
    raise RuntimeError(f"cross-config inference dir not found: {_CC}")
for p in (_WF, _CC, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import predict_cross_config as cc                                    # noqa: E402
from dump_dvfs_forecast import build_args, walk_forward, TARGET      # noqa: E402

FREQS = [1.0, 2.0, 3.0, 4.0]
CORES = {"P": "0", "E": "16"}
CONFIGS = [f"{p}_{f:.1f}GHz" for p in ("P", "E") for f in FREQS]


def cfg_parts(cfg):
    """'P_3.0GHz' -> ('0', '3.0')"""
    m = re.match(r"([PE])_([0-9.]+)GHz$", cfg)
    if not m:
        raise ValueError(f"unparseable config: {cfg}")
    return CORES[m.group(1)], m.group(2)


def load_all_traces(bench):
    """Full concatenated trace per configuration. Missing any one is fatal: a stitched
    history cannot be built for a path that visits a configuration we cannot read."""
    out = {}
    for cfg in CONFIGS:
        cpu, freq = cfg_parts(cfg)
        t = cc.load_full_trace(cc.workload_name(bench, freq, cpu))
        if t is None:
            raise SystemExit(f"missing trace for {bench} {cfg}")
        out[cfg] = t.reset_index(drop=True)
    return out


def phase_lengths(bench):
    """[(phase_num, nrows), ...] in the concat order load_full_trace produces."""
    cpu, freq = cfg_parts("P_3.0GHz")
    wl = cc.workload_name(bench, freq, cpu)
    single = glob.glob(os.path.join(cc.DATA_DIR, "**", f"{wl}.csv"), recursive=True)
    if single:
        return [(0, len(pd.read_csv(single[0], usecols=[0])))]
    by_num = {}
    for p in sorted(glob.glob(os.path.join(cc.DATA_DIR, "**", f"{wl}_phase*.csv"), recursive=True)):
        by_num.setdefault(int(re.search(r"_phase(\d+)\.csv$", p).group(1)), p)
    return [(n, len(pd.read_csv(by_num[n], usecols=[0]))) for n in sorted(by_num)]


def load_path(actions_dir, bench, metric, policy, phases, total):
    """Concatenated per-chunk configuration across phases, in trace order.

    The simulator runs each workload-phase independently, so the dumped chunk indices are
    phase-local; they are concatenated here in the same order as the traces.
    """
    path = []
    for ph, seglen in phases:
        f = os.path.join(actions_dir, f"{bench}__{ph}__{metric}__{policy}.csv")
        if not os.path.exists(f):
            raise SystemExit(f"missing action dump: {f}")
        d = pd.read_csv(f)
        if not {"chunk", "config"} <= set(d.columns):
            raise SystemExit(f"action dump lacks chunk/config columns: {f}")
        cfgs = d.sort_values("chunk")["config"].tolist()
        if len(cfgs) < seglen:
            # The sim may drop trailing chunks; carry the last decision forward rather than
            # inventing one, and never pad a phase we have no decisions for at all.
            if not cfgs:
                raise SystemExit(f"empty action dump: {f}")
            cfgs = cfgs + [cfgs[-1]] * (seglen - len(cfgs))
        path.extend(cfgs[:seglen])
    if len(path) != total:
        raise SystemExit(f"path length {len(path)} != trace length {total} for {bench}")
    bad = sorted(set(path) - set(CONFIGS))
    if bad:
        raise SystemExit(f"action dump contains unknown configs {bad} for {bench}")
    return path


def runs_of(path):
    """[(lo, hi, config), ...] maximal runs of constant configuration."""
    out, lo = [], 0
    for i in range(1, len(path) + 1):
        if i == len(path) or path[i] != path[lo]:
            out.append((lo, i, path[lo]))
            lo = i
    return out


def stitch(traces, bench, path, target, counters):
    """Counter frame for `target` assembled along `path`.

    Each run is translated from the configuration actually occupied. translate_counter is
    the identity when source == target, so runs already on the target contribute their raw
    measurements with no model in the loop.
    """
    t_cpu, t_freq = cfg_parts(target)
    n = len(path)
    cols = {c: np.empty(n, dtype=float) for c in counters}
    n_model = 0
    for lo, hi, src in runs_of(path):
        s_cpu, s_freq = cfg_parts(src)
        seg = traces[src].iloc[lo:hi]
        for c in counters:
            if c not in seg.columns:
                raise SystemExit(f"counter {c} missing from {bench} {src}")
            if c in (TARGET, "cpu_cycles"):
                v = cc.translate_counter(seg, bench, c, s_cpu, s_freq, t_cpu, t_freq)
                if v is None:
                    raise SystemExit(
                        f"no {c} translator {src}->{target} for {bench}; refusing to emit "
                        f"an unstitched history")
            else:
                # branches / instructions are config-invariant (instructions is pinned at
                # 10M per chunk by construction), so the occupied config's own values stand.
                v = seg[c].values
            cols[c][lo:hi] = np.asarray(v, dtype=float)
        if src != target:
            n_model += hi - lo
    return pd.DataFrame(cols), n_model


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--actions_dir", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--metric", required=True, choices=["EDP", "ED2P"])
    ap.add_argument("--out_proc", required=True, help="cross-processor output dir")
    ap.add_argument("--out_freq", required=True, help="cross-frequency output dir")
    ap.add_argument("--counters", nargs="+",
                    default=["ref_cycles", "cpu_cycles", "branches", "instructions"])
    ap.add_argument("--model", default="dt")
    ap.add_argument("--classifier", default="gmm")
    ap.add_argument("--phase_count", type=int, default=6)
    ap.add_argument("--timesteps", type=int, default=5)
    ap.add_argument("--filter_size", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--method", default="per_phase", choices=["global", "per_phase"])
    ap.add_argument("--gate", default="none", choices=["none", "phase", "persist", "both"])
    ap.add_argument("--gate_window", type=int, default=200)
    ap.add_argument("--gate_margin", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=2048)
    ap.add_argument("--block", type=int, default=4096)
    ap.add_argument("--max_train", type=int, default=20000)
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--emit", default="both", choices=["proc", "freq", "both"],
                    help="Which half to write. The two halves are translated with DIFFERENT "
                         "model variants in the baseline (cross-proc general_temporal, "
                         "cross-freq top4), and CBM_FEATURE_SET is global, so producing a "
                         "comparable pair means two invocations -- one per half, each with its "
                         "matching --feature_set. Emitting both from one run would translate "
                         "one half with the wrong variant.")
    ap.add_argument("--feature_set", required=True,
                    help="Translator variant this dump must use, e.g. general_temporal. It is "
                         "read from CBM_FEATURE_SET at import time and defaults to top4, so a "
                         "caller that forgets to export it silently translates with a different "
                         "model than the baseline it will be compared against. Passing it here "
                         "turns that into an error.")
    a = ap.parse_args()

    import src.create_dataset as cds
    active = getattr(cds, "CBM_FEATURE_SET", None)
    if active != a.feature_set:
        raise SystemExit(
            f"CBM_FEATURE_SET is '{active}' but --feature_set is '{a.feature_set}'. It is read "
            f"at import, so export it before starting python: "
            f"CBM_FEATURE_SET={a.feature_set} {' '.join(sys.argv[:1])} ...")
    print(f"translator feature set: {active}", flush=True)

    args = build_args(a.counters, a.model, a.timesteps, a.classifier, a.phase_count,
                      a.epochs, a.filter_size, gate_window=a.gate_window,
                      gate_margin=a.gate_margin)

    traces = load_all_traces(a.bench)
    phases = phase_lengths(a.bench)
    lens = {c: len(t) for c, t in traces.items()}
    # Configurations do not always land on exactly the same chunk count (the E-core traces
    # commonly run one short), so work to the shortest and trim the final phase to match.
    total = min(min(lens.values()), sum(n for _, n in phases))
    trimmed, acc = [], 0
    for ph, n in phases:
        if acc >= total:
            break
        take = min(n, total - acc)
        trimmed.append((ph, take))
        acc += take
    phases = trimmed
    if acc != total:
        raise SystemExit(f"cannot align {a.bench}: phases give {acc}, traces give {total}")

    path = load_path(a.actions_dir, a.bench, a.metric, a.policy, phases, total)
    switches = sum(1 for i in range(1, len(path)) if path[i] != path[i - 1])

    # One forecast per target configuration. The source axis has collapsed: the stitched
    # history already encodes where the workload was.
    # Only the targets the emitted half needs. proc = the other core's four configs,
    # freq = this core's other three, for every source. Across all sources that is still every
    # configuration, so the loop is unchanged -- but keeping it explicit documents why.
    fc_time, base_time = {}, {}
    for tgt in CONFIGS:
        frame, n_model = stitch(traces, a.bench, path, tgt, a.counters)
        raw = frame[TARGET].values.astype(float)
        base = np.empty(total)
        base[1:] = raw[:-1]
        base[0] = raw[0]
        base_time[tgt] = base / 1e9                      # stitched persistence, the fallback
        tt = base_time[tgt].copy()
        pred = walk_forward(args, frame, a.method, a.warmup, a.block, a.max_train,
                            a.horizon, gate=a.gate)
        for lab, v in pred.items():
            if 0 <= lab < total:
                tt[lab] = v / 1e9
        fc_time[tgt] = tt
        print(f"  {tgt}: {n_model}/{total} chunks translated "
              f"({n_model / total * 100:.1f}%), {len(pred)} forecast", flush=True)

    if not fc_time:
        raise SystemExit(f"no targets forecast for {a.bench}")

    # Emit into every source slice. Speedup is written against that slice's own carrier so
    # each decodes to the same stitched target time.
    for src in CONFIGS:
        s_pre = src[0]
        s_ghz = src.split("_", 1)[1]
        time_src = traces[src][TARGET].values[:total].astype(float) / 1e9
        halves = {"proc": (True,), "freq": (False,), "both": (True, False)}[a.emit]
        for is_xp in halves:
            t_pre = ("E" if s_pre == "P" else "P") if is_xp else s_pre
            tgts = [f"{t_pre}_{f:.1f}GHz" for f in FREQS]
            if not is_xp:
                tgts = [t for t in tgts if t != src]
            out_dir = a.out_proc if is_xp else a.out_freq
            sub = os.path.join(out_dir, f"speedups_from_{src}")
            os.makedirs(sub, exist_ok=True)
            r0 = 0
            for ph, seglen in phases:
                sl = slice(r0, r0 + seglen)
                ts = time_src[sl]
                cols = {"sample_index": np.arange(seglen), f"Time_{src}": ts}
                for tgt in tgts:
                    ft = fc_time[tgt][sl]
                    spd = ts / np.where(ft > 0, ft, np.nan)
                    ref = ts / np.where(base_time[tgt][sl] > 0, base_time[tgt][sl], np.nan)
                    if is_xp:
                        # Same guard as dump_dvfs_forecast: cross-proc speedup has no
                        # frequency-ratio physics, so bound forecast blowups by the
                        # stitched-persistence reference rather than a single source's.
                        spd = np.clip(spd, ref * 0.5, ref * 2.0)
                    else:
                        tf = float(tgt.split("_")[1].replace("GHz", ""))
                        sf = float(s_ghz.replace("GHz", ""))
                        lo_r, hi_r = min(1.0, tf / sf), max(1.0, tf / sf)
                        spd = np.clip(spd, lo_r * 0.95, hi_r * 1.05)
                    cols[f"Speedup_{tgt}_vs_{src}"] = spd
                fname = (f"{a.bench}_phase{ph}.csv" if is_xp
                         else f"speedups_{src}_{a.bench}_phase{ph}.csv")
                pd.DataFrame(cols).to_csv(os.path.join(sub, fname), index=False)
                r0 += seglen
    print(f"wrote {a.bench}: {total} chunks, {switches} config switches "
          f"({switches / max(total, 1) * 100:.3f}%), policy={a.policy} metric={a.metric}",
          flush=True)


if __name__ == "__main__":
    main()

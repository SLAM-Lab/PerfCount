#!/usr/bin/env python3
"""Chapter 5 result verifier.

Recomputes every backed-by-data number in Chapter 5 from the canonical artifacts
(symlinked under inputs/) and compares each against the value currently written in
the chapter tex. Prints one line per check: OK / STALE / (info), with the tex value,
the recomputed value, and the tex location.

Sources of truth
  scheduling  : inputs/scheduling/{hetero,DVFS}/{general,loocv,perfectcp}/all_phases_summary.csv
                (normalized to the global Viterbi oracle; reuses report_chapter5.py loaders)
  forecasting : inputs/forecast_condensed/het_cross_*_condensed_10M_cpu*.csv  + inputs/forecast_baseline/baseline_p0_cpu*.csv
  inference   : inputs/inference_cross_config/cross_config_10M_modelcmp{,_refcpu}.csv  (medians; MAPE is outlier-heavy)

Nothing here retrains a model; every number is a re-aggregation of dumped results,
so a full run is seconds. Run:  ./run.sh   (or: <venv>/python3 verify_ch5.py)
"""
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
INP = os.path.join(HERE, "inputs")
SCHED_SRC = os.path.join(HERE, "..", "scheduling")
sys.path.insert(0, os.path.abspath(SCHED_SRC))

# Reuse the canonical scheduling loaders/constants so this can never drift from the
# number the chapter's own reproduction script prints.
os.environ.setdefault("RES", os.path.join(INP, "scheduling"))
import report_chapter5 as R  # noqa: E402

TEX = open(os.path.join(INP, "chapter5.tex")).read()
SUITES = ["SPEC2017", "SPEC2026", "DaCapo"]


def tex_row(label, rowkey):
    """Parse the numeric cells of one table row directly from the tex, so table checks
    track the live tex instead of a hardcoded copy. Returns the floats after `rowkey`."""
    i = TEX.find("\\label{" + label + "}")
    if i < 0:
        return None
    m = re.search(re.escape(rowkey) + r"\s*&(.+?)\\\\", TEX[i:i + 2600], re.S)
    return [float(x) for x in re.findall(r"\d+\.\d+", m.group(1))] if m else None

# ------------------------------------------------------------------ reporting
_tot = {"ok": 0, "stale": 0, "info": 0}
_stale = []


def _fmt(v):
    return "  --  " if v is None or (isinstance(v, float) and v != v) else f"{v:.3f}"


def check(label, tex_val, got, tol=0.006, loc=""):
    """Compare a computed value against what the tex says. tex_val=None -> info only."""
    if tex_val is None:
        _tot["info"] += 1
        print(f"  info  {label:<52} data={_fmt(got)}   {loc}")
        return
    ok = got == got and abs(got - tex_val) <= tol
    _tot["ok" if ok else "stale"] += 1
    tag = " OK " if ok else "STALE"
    line = f"  {tag}  {label:<52} tex={_fmt(tex_val)} data={_fmt(got)}   {loc}"
    print(line)
    if not ok:
        _stale.append((label, tex_val, got, loc))


def head(t):
    print("\n" + "=" * 92 + f"\n{t}\n" + "=" * 92)


# ------------------------------------------------------------------ scheduling helpers
def het(arm, s, metric="EDP"):
    return R.load_het(arm, metric, s)


def het_spec(arm, metric="EDP"):
    """Pool both SPEC suites (the 120 workload-phases the gate-rescue table reports)."""
    f = os.path.join(R.RES, "hetero", R.HET[arm], "all_phases_summary.csv")
    e = pd.read_csv(f)
    e["S"] = e.Workload.map(R.suite)
    e = e[(~e.Workload.isin(R.EXC)) & (e.Metric == metric) & (e.S != "DaCapo")]
    return e.pivot_table(index=["Workload", "Phase"], columns="Policy", values="Final_Value")


HEUR_LABEL = [("EAS", "EAS_Hetero"), ("EAS+su", "EAS_With_DVFS"), ("uEAS", "Micro_EAS"),
              ("Thresh", "Threshold_Migration"), ("ITD", "Thread_Director"), ("UCB1", "UCB1_Hetero")]


# ------------------------------------------------------------------ 1. het prior table (tab:het_prior)
def check_het_prior():
    head("HET PRIOR HEURISTICS  (tab:het_prior, L1025-1027; General arm, EDP, /global oracle)")
    texkey = {"SPEC2017": "SPEC17", "SPEC2026": "SPEC26", "DaCapo": "DaCapo"}
    for s in SUITES:
        g = het("General", s)
        vals = [R.r(g, p) for _, p in HEUR_LABEL] + [R.r(g, "EAS_Oracle_Hetero"), R.r(g, "Thread_Director_Oracle")]
        labs = [l for l, _ in HEUR_LABEL] + ["EAS-PF", "ITD-PF"]
        row = tex_row("tab:het_prior", texkey[s]) or [None] * 8
        for lab, got, tv in zip(labs, vals, row):
            check(f"{s} {lab}", tv, got, loc="tab:het_prior")


# ------------------------------------------------------------------ 2. het ladder (tab:het_ladder)
def check_het_ladder():
    head("HET LADDER  (tab:het_ladder, L1065-1067; EDP, /global oracle)")
    texkey = {"SPEC2017": "SPEC17", "SPEC2026": "SPEC26", "DaCapo": "DaCapo"}
    for s in SUITES:
        g = het("General", s)
        pcp = het("Perfect-CP", s)
        bh = min(R.r(g, p) for _, p in HEUR_LABEL)
        got = [bh, R.r(g, R.REACT), R.r(g, R.GATE), R.r(pcp, R.GATE), R.r(g, R.GREEDY_TRUEP)]
        row = (tex_row("tab:het_ladder", texkey[s]) or [None] * 6)[:5]  # drop the oracle 1.000 cell
        for lab, gv, tv in zip(["best-heur", "reactive", "gate", "perfect-CP", "greedy+trueP"], got, row):
            check(f"{s} {lab}", tv, gv, loc="tab:het_ladder")


# ------------------------------------------------------------------ 3. gate-rescue "all" rows (tab:ch5_gate_rescue)
def check_gate_rescue():
    head("GATE-RESCUE all-SPEC rows  (tab:ch5_gate_rescue, L1128/L1133; /global oracle, 120 phases)")
    TEXVAL = {  # reactive, forecast, gated, forecast-worse/120, gate-worse/120
        "EDP":  [1.121, 1.144, 1.115, 96, 70],
        "ED2P": [1.194, 1.289, 1.194, 92, 65],
    }
    for metric in ["EDP", "ED2P"]:
        p = het_spec("General", metric)
        rr, ff, gg = R.r(p, R.REACT), R.r(p, R.RAWFC), R.r(p, R.GATE)
        fw = int((p[R.RAWFC] > p[R.REACT]).sum())
        gw = int((p[R.GATE] > p[R.REACT]).sum())
        tv = TEXVAL[metric]
        check(f"{metric} reactive", tv[0], rr, loc="tab:ch5_gate_rescue")
        check(f"{metric} forecast", tv[1], ff, loc="tab:ch5_gate_rescue")
        check(f"{metric} gated", tv[2], gg, loc="tab:ch5_gate_rescue")
        check(f"{metric} forecast-worse/120", tv[3], fw, tol=0.5, loc="tab:ch5_gate_rescue")
        check(f"{metric} gate-worse/120", tv[4], gw, tol=0.5, loc="tab:ch5_gate_rescue")


# ------------------------------------------------------------------ 4. headline (prose L50/L1320/L1590)
def check_headline():
    head("HEADLINE gate vs best heuristic  (General, EDP; prose L50, L1320, L1590)")
    TEXVAL = {"SPEC2017": 21.6, "SPEC2026": 18.6, "DaCapo": 21.1}
    for s in SUITES:
        g = het("General", s)
        gate = R.r(g, R.GATE)
        bh = min(R.r(g, p) for _, p in HEUR_LABEL)
        win = (bh - gate) / bh * 100
        check(f"{s} gate wins by %", TEXVAL[s], win, tol=0.15, loc="prose L50/1320/1590")


# ------------------------------------------------------------------ 5. where forecasting pays (tab:het_split)
def check_het_split():
    head("WHERE FORECASTING PAYS  (tab:het_split L1245-1254; gate over reactive, SPEC het/homo)")
    # canonical from report_chapter5.het_split(): General EDP het +0.96; Perfect-CP EDP het +1.19, ED2P +2.20
    print("  (recompute via report_chapter5.het_split — printed below for reference)")
    R.het_split()


# ------------------------------------------------------------------ 6. DVFS ladder + gate (regen source)
DVFS_RAW = os.path.join(R.RES, "DVFS_Study", "dvfs_raw", "all_phases_summary.csv")


def check_dvfs():
    head("DVFS gate table  (tab:ch5_dvfs_gate, L673-677) vs the tex's source-of-record DVFS_Study/dvfs_raw")
    # tex reports the DVFS tables from DVFS_Study/dvfs_raw (PROV L555/595/628), NOT report_chapter5's
    # DVFS/general -- so verify against dvfs_raw, the arm the chapter is actually built on.
    TEX = {("P", "EDP"): (1.049, 1.042, 0.68), ("P", "ED2P"): (1.078, 1.075, 0.27),
           ("E", "EDP"): (1.062, 1.041, 1.96), ("E", "ED2P"): (1.071, 1.070, 0.12)}
    e = pd.read_csv(DVFS_RAW)
    e = e[(~e.Workload.isin(R.EXC)) & (e.Workload.map(R.suite) != "DaCapo")]
    for c in ["P", "E"]:
        for metric in ["EDP", "ED2P"]:
            p = e[e.Metric == metric].pivot_table(index=["Workload", "Phase"], columns="Policy", values="Final_Value")
            n = f"Global_Oracle_{c}"
            rr = (p[f"Model_Greedy_{c}"] / p[n]).mean()
            gg = (p[f"Model_Forecast_ReactiveGated_{c}"] / p[n]).mean()
            adv = ((p[f"Model_Greedy_{c}"] - p[f"Model_Forecast_ReactiveGated_{c}"]) / p[f"Model_Greedy_{c}"] * 100).mean()
            tr, tg, ta = TEX[(c, metric)]
            check(f"{c}-core {metric} reactive", tr, rr, loc="tab:ch5_dvfs_gate")
            check(f"{c}-core {metric} gated", tg, gg, loc="tab:ch5_dvfs_gate")
            check(f"{c}-core {metric} gate-vs-react %", ta, adv, tol=0.05, loc="tab:ch5_dvfs_gate")
    print("\n  NOTE: report_chapter5.py reads DVFS/general, which gives DIFFERENT DVFS numbers than")
    print("  dvfs_raw (e.g. E-core EDP gate advantage +0.26% vs the tex's +1.96%). The tex is on")
    print("  dvfs_raw per its PROV notes; this check validates that. If DVFS/general is meant to be")
    print("  canonical instead, the DVFS tables need regenerating (regen_dvfs_tables.py).")


# ------------------------------------------------------------------ 7. forecasting under heterogeneity
def _base_pm(cpu):
    d = pd.read_csv(os.path.join(INP, "forecast_baseline", f"baseline_p0_cpu{cpu}.csv"))
    d = d[d.horizon == 1]
    return {m: d[d.model == m].groupby("workload").baseline.mean().mean() for m in ["dt", "mlp", "lstm", "transformer"]}


def _cond_pm(problem, tag, cpu, p):
    f = os.path.join(INP, "forecast_condensed", f"het_cross_{problem}_{tag}_condensed_10M_cpu{cpu}.csv")
    d = pd.read_csv(f)
    d = d[(d.horizon == 1) & (d.timesteps == 5) & (d.het_prob == p)]
    return {m: d[d.model == m].groupby("workload").mape.mean().mean() for m in ["dt", "mlp", "lstm", "transformer"]}


def _rng(d):
    v = list(d.values())
    return min(v), max(v)


def check_forecasting():
    head("FORECASTING UNDER HETEROGENEITY  (cross-freq L108/112, cross-proc L128/130)")
    # cross-frequency (cpu0)
    b = _base_pm(0)
    nv = _cond_pm("freq", "naive", 0, 1.0)
    tr = _cond_pm("freq", "translated", 0, 1.0)
    check("cross-freq baseline lo", 9, round(_rng(b)[0]), tol=0.6, loc="L108")
    check("cross-freq baseline hi", 12, round(_rng(b)[1]), tol=0.6, loc="L108")
    check("cross-freq naive lo", 50, round(_rng(nv)[0]), tol=1.5, loc="L108")
    check("cross-freq naive hi", 83, round(_rng(nv)[1]), tol=1.5, loc="L108")
    check("cross-freq translated lo", 9, round(_rng(tr)[0]), tol=1, loc="L112")
    check("cross-freq translated hi", 14, round(_rng(tr)[1]), tol=1, loc="L112")
    # cross-processor: pool both cores (P from E = cpu0, E from P = cpu16)
    for cpu, dirn in [(0, "P<-E"), (16, "E<-P")]:
        b2 = _base_pm(cpu)
        nv2 = _cond_pm("proc", "naive", cpu, 1.0)
        tr2 = _cond_pm("proc", "translated", cpu, 1.0)
        check(f"cross-proc {dirn} naive range", None, _rng(nv2)[1], loc="L128")
        check(f"cross-proc {dirn} translated range", None, _rng(tr2)[1], loc="L130")
    # combined ranges the prose states
    nvP, nvE = _cond_pm("proc", "naive", 0, 1.0), _cond_pm("proc", "naive", 16, 1.0)
    trP, trE = _cond_pm("proc", "translated", 0, 1.0), _cond_pm("proc", "translated", 16, 1.0)
    both_nv = min(_rng(nvP)[0], _rng(nvE)[0]), max(_rng(nvP)[1], _rng(nvE)[1])
    both_tr = min(_rng(trP)[0], _rng(trE)[0]), max(_rng(trP)[1], _rng(trE)[1])
    check("cross-proc naive lo (both dirs)", 33, round(both_nv[0]), tol=1.5, loc="L128")
    check("cross-proc naive hi (both dirs)", 99, round(both_nv[1]), tol=1.5, loc="L128")
    check("cross-proc translated lo", 11, round(both_tr[0]), tol=1, loc="L130")
    check("cross-proc translated hi", 19, round(both_tr[1]), tol=1, loc="L130")


# ------------------------------------------------------------------ 8. inference (medians; MAPE outlier-heavy)
def check_inference():
    head("INFERENCE-TIME (translate-then-forecast)  medians; L139-158")
    cc = os.path.join(INP, "inference_cross_config")
    refcpu = pd.read_csv(os.path.join(cc, "cross_config_10M_modelcmp_refcpu.csv"))
    ref_f = os.path.join(cc, "cross_config_10M_modelcmp.csv")
    ref = pd.read_csv(ref_f)
    nmod = ref.groupby("model").bench.nunique().to_dict()
    complete = all(nmod.get(m, 0) >= 68 for m in ["dt", "mlp", "lstm", "transformer"])
    if not complete:
        print(f"  !! ref_cycles arm (modelcmp.csv) INCOMPLETE: benches/model={nmod} -- ref-only numbers provisional\n")
    freq = lambda d: (d.src_cpu == d.tgt_cpu) & (d.src_freq != d.tgt_freq)
    proc = lambda d: (d.src_cpu != d.tgt_cpu)
    med = lambda d, m, mk: d[(d.method == m) & mk(d)].mape.median()
    # translation-independent (verifiable from complete refcpu arm)
    check("oracle median", 13.9, refcpu[refcpu.method == "oracle"].mape.median(), tol=1.0, loc="L139")
    check("naive cross-freq", 37.2, med(refcpu, "naive", freq), tol=2.5, loc="L142")
    check("naive cross-proc", 46.2, med(refcpu, "naive", proc), tol=2.5, loc="L142")
    # +cpu translated (verifiable now)
    check("+cpu translated cross-proc", 20.0, med(refcpu, "translated", proc), tol=2.0, loc="L148")
    # ref-only translated (needs complete arm1)
    tv = None if not complete else 15.4
    check("ref-only translated cross-freq", tv, med(ref, "translated", freq), tol=2.0, loc="L146")


def main():
    check_het_prior()
    check_het_ladder()
    check_gate_rescue()
    check_headline()
    check_forecasting()
    check_inference()
    check_dvfs()      # print-only canonical values for the regeneration
    check_het_split()  # print-only reference

    head("SUMMARY")
    print(f"  OK: {_tot['ok']}    STALE: {_tot['stale']}    info-only: {_tot['info']}")
    if _stale:
        print("\n  STALE (tex disagrees with data):")
        for lab, tv, gv, loc in _stale:
            print(f"    - {loc:<22} {lab:<40} tex={_fmt(tv)} -> should be {_fmt(gv)}")
    print("\n  Note: DVFS tables in tex are on the OLD dvfs_raw power; the DVFS block above prints the")
    print("  canonical refined-power values to regenerate them from. Inference ref-only numbers are")
    print("  provisional until the modelcmp.csv (ref_cycles) arm finishes re-running.")


if __name__ == "__main__":
    main()

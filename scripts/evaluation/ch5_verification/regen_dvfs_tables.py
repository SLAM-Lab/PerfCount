#!/usr/bin/env python3
"""Regenerate the DVFS table cells on the canonical REFINED power (DVFS/general, DVFS/loocv),
which the chapter's DVFS tables currently do NOT use (they are on the old DVFS_Study/dvfs_raw).

Emits LaTeX-ready rows for tab:dvfs_gov_edp, tab:dvfs_gov_ed2p, tab:dvfs_ladder, and the
tab:ch5_dvfs_gate cells. Every value is normalized to the per-core Global_Oracle, marian and
stockfish excluded, per-phase suite means -- the same conventions report_chapter5.py uses.

Run:  <venv>/python3 regen_dvfs_tables.py
"""
import os
import sys

import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "scheduling")))
os.environ.setdefault("RES", os.path.join(HERE, "inputs", "scheduling"))
import report_chapter5 as R  # noqa

SUITE_TEX = [("SPEC2017", "SPEC17"), ("SPEC2026", "SPEC26"), ("DaCapo", "DaCapo")]
GOV = [("Perf", "Performance_Gov"), ("PwrSv", "Powersave_Gov"), ("OnDem", "Ondemand"),
       ("Cons", "Conservative"), ("Sched", "Schedutil_PELT"), ("Intr", "Interactive"),
       ("EWMA", "EWMA"), ("HWP", "Intel_HWP"), ("UCB1", "UCB1")]
PF_GOV = [("OnDem", "Ondemand_Future"), ("Cons", "Conservative_Future"), ("Sched", "Schedutil_Future"),
          ("Intr", "Interactive_Future"), ("EWMA", "EWMA_Future"), ("HWP", "Intel_HWP_Future")]
FREQS = ["1.0", "2.0", "3.0", "4.0"]


def load(metric, arm="General"):
    f = os.path.join(R.RES, "DVFS", R.DVFS[arm], "all_phases_summary.csv")
    e = pd.read_csv(f)
    e = e[(~e.Workload.isin(R.EXC)) & (e.Metric == metric)]
    e["S"] = e.Workload.map(R.suite)
    return e


def per_suite(e, s, c):
    p = e[e.S == s].pivot_table(index=["Workload", "Phase"], columns="Policy", values="Final_Value")
    norm = f"Global_Oracle_{c}"
    return (lambda pol: (p[pol] / p[norm]).mean() if pol in p.columns else float("nan")), p


def f3(x):
    return "--" if x != x else f"{x:.3f}"


def gov_table(metric):
    print(f"\n%% ---- tab:dvfs_gov_{metric.lower()}  (REFINED power) ----")
    e = load(metric)
    for c in ["P", "E"]:
        for s, stex in SUITE_TEX:
            col, _ = per_suite(e, s, c)
            react = [col(f"{pol}_{c}") for _, pol in GOV]
            pf = [col(f"{pol}_{c}") for _, pol in PF_GOV]
            static = min((col(f"Static_{c}_{fq}GHz") for fq in FREQS), default=float("nan"))
            cells = " & ".join(f3(x) for x in react) + " & & " + " & ".join(f3(x) for x in pf) + f" & & {f3(static)}"
            print(f" & {stex} & {cells} \\\\")
        print("\\midrule" if c == "P" else "")


def ladder():
    print("\n%% ---- tab:dvfs_ladder (EDP, REFINED power). Forecasting-True col flagged ----")
    e = load("EDP")
    for c in ["P", "E"]:
        for s, stex in SUITE_TEX:
            col, _ = per_suite(e, s, c)
            static = min((col(f"Static_{c}_{fq}GHz") for fq in FREQS), default=float("nan"))
            proxy = min((col(f"{pol}_{c}") for _, pol in GOV))       # best reactive governor
            model = col(f"Model_Greedy_{c}")
            rtrue = col(f"Reactive_Oracle_{c}")
            raw = col(f"Model_Forecast_{c}")
            gated = col(f"Model_Forecast_ReactiveGated_{c}")
            pf = col(f"Model_Greedy_Oracle_{c}")
            gap = model - pf
            print(f" & {stex} & {f3(static)} & {f3(proxy)} & {f3(model)} & {f3(rtrue)} & "
                  f"{f3(raw)} & {f3(gated)} & \\tbd & {f3(pf)} & {gap:.3f} \\\\   %% Forecasting-True(=Forecast_Oracle) needs refined fctrue arm")
        print("\\midrule" if c == "P" else "")


def gate_table():
    print("\n%% ---- tab:ch5_dvfs_gate (SPEC-pooled per core, REFINED power) ----")
    print("%% core metric : reactive forecast gated PF  gate-vs-reactive(paired mean, p)")
    for c in ["P", "E"]:
        for metric in ["EDP", "ED2P"]:
            e = load(metric)
            p = e[e.S != "DaCapo"].pivot_table(index=["Workload", "Phase"], columns="Policy", values="Final_Value")
            norm = f"Global_Oracle_{c}"
            col = lambda pol: (p[pol] / p[norm])
            react, fore = col(f"Model_Greedy_{c}"), col(f"Model_Forecast_{c}")
            gate, pf = col(f"Model_Forecast_ReactiveGated_{c}"), col(f"Model_Greedy_Oracle_{c}")
            adv = ((react - gate) / react * 100).mean()
            pv = stats.wilcoxon(react.values, gate.values).pvalue
            print(f" {c} & {metric} & {react.mean():.3f} & {fore.mean():.3f} & {gate.mean():.3f} & "
                  f"{pf.mean():.3f} & ${adv:+.2f}$\\% ($p={pv:.3f}$) \\\\")


if __name__ == "__main__":
    gov_table("EDP")
    gov_table("ED2P")
    ladder()
    gate_table()

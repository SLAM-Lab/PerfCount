#!/usr/bin/env python3
# Print a comparison table across model runs produced by run_hetero.sh / run_dvfs.sh.
# Usage:  RES=<results/scheduling> python compare_runs.py <hetero|dvfs> <model> [<model> ...]
import os, sys, pandas as pd

RES = os.environ.get("RES", "results/scheduling")
mode, models = sys.argv[1], sys.argv[2:]

def summary(prefix, model, metric):
    df = pd.read_csv(f"{RES}/{prefix}/{model}/all_phases_summary.csv")
    df = df[df.Metric == metric]
    return df.pivot_table(index=["Workload", "Phase"], columns="Policy", values="Final_Value")

def ratio(E, pol, norm):
    return (E[pol] / E[norm]).mean() if pol in E and norm in E else float("nan")

if mode == "hetero":
    # reactive / forecast / gate, normalized to the heterogeneous oracle
    for metric in ["EDP", "ED2P"]:
        print(f"\n  {metric:5s}  {'model':13s} {'reactive':>9s} {'forecast':>9s} {'gate':>7s}")
        for m in models:
            E = summary("hetero", m, metric)
            r = ratio(E, "Model_Reactive_Hetero", "Proactive_Hetero_Oracle")
            f = ratio(E, "Model_Forecast_Hetero", "Proactive_Hetero_Oracle")
            g = ratio(E, "Model_Forecast_ReactiveGated_Hetero", "Proactive_Hetero_Oracle")
            print(f"         {m:13s} {r:9.3f} {f:9.3f} {g:7.3f}")

else:  # dvfs: reactive / forecast / perfect-future per core, normalized to the per-core oracle
    for core in ["P", "E"]:
        for metric in ["EDP", "ED2P"]:
            print(f"\n  {core}-core {metric:5s}  {'model':13s} {'reactive':>9s} {'forecast':>9s} {'perfect-fut':>12s}")
            for m in models:
                E = summary("DVFS", m, metric)
                n = f"Global_Oracle_{core}"
                r = ratio(E, f"Model_Greedy_{core}", n)
                f = ratio(E, f"Model_Forecast_{core}", n)
                pf = ratio(E, f"Model_Greedy_Oracle_{core}", n)
                print(f"                {m:13s} {r:9.3f} {f:9.3f} {pf:12.3f}")

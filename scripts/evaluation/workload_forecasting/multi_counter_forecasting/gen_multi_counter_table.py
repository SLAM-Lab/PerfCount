"""Multi-counter forecasting table (Chapter 4).

For each top cross-platform counter, how forecastable is it one window ahead?
Reports median MAPE for DT and MLP on both x86 cores, alongside the counter's
coefficient of variation (CoV) computed on the SAME concatenated trace the
forecaster reads, so the "dense forecasts, bursty does not" story is anchored to
a measured variability statistic rather than an asserted one.

Reads the condensed sweep CSVs beside this script; CoV from processed_data_10M.
Emits multi_counter_forecast.tex + prints the prose-relevant numbers.

Usage:  python gen_multi_counter_table.py
"""
import os
import re
import glob
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "../../../.."))
DATA = os.path.join(REPO, "processed_data_10M", "x86_desktop_heterogeneous")
OUT = os.path.join(HERE, "multi_counter_forecast.tex")

# Order: the two dense counters first (the forecastable class), then miss-events
# by ascending CoV. Names shown in the table match the importance tables.
COUNTERS = ["branches", "ref_cycles", "branch_misses", "branch_load_misses",
            "cache_references", "dtlb_load_misses", "l1_icache_load_misses",
            "cache_misses", "llc_loads", "llc_misses"]
TEX = {c: c.replace("_", "\\_") for c in COUNTERS}


def workload_cov(counter, cpu, freq="4.0"):
    """Median over workloads of each concatenated trace's CoV for `counter`,
    matching how the forecaster assembles a trace (phase files concatenated)."""
    covs = []
    benches = {}
    for f in glob.glob(os.path.join(DATA, "*", "*", f"aligned_spec_*_{freq}GHz_cpu{cpu}_phase*.csv")):
        b = re.sub(r"_phase\d+\.csv$", "", os.path.basename(f))
        benches.setdefault(b, []).append(f)
    for b, fs in benches.items():
        try:
            s = pd.concat([pd.read_csv(f, usecols=[counter])[counter] for f in sorted(fs)],
                          ignore_index=True).astype(float)
        except Exception:
            continue
        if s.mean() > 0:
            covs.append(s.std() / s.mean())
    return np.median(covs) if covs else float("nan")


def load(cpu):
    d = pd.read_csv(os.path.join(HERE, f"condensed_multi_counter_cpu{cpu}.csv"))
    return d[d.status == "Success"]


def main():
    stats = {}
    for cpu in (0, 16):
        d = load(cpu)
        for c in COUNTERS:
            row = stats.setdefault(c, {})
            row[f"cov{cpu}"] = workload_cov(c, cpu)
            for m in ("dt", "mlp"):
                q = d[(d.target == c) & (d.model == m)].mape
                row[f"{m}{cpu}"] = q.median() if len(q) else float("nan")

    def cell(x):
        return "--" if (x is None or (isinstance(x, float) and np.isnan(x))) else f"{x:.1f}"

    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Single-step ($H{=}1$) forecast MAPE (\%) for the top cross-platform "
        r"counters on the two x86 cores, median over the SPEC suites for the DT and MLP "
        r"forecasters, with each counter's coefficient of variation (CoV) on the same trace. "
        r"Only the two dense, low-CoV counters forecast well.}",
        r"\label{tab:multi_counter_forecast}",
        r"\small", r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{l ccc c ccc}", r"\toprule",
        r"& \multicolumn{3}{c}{\textbf{P-core}} & & \multicolumn{3}{c}{\textbf{E-core}} \\",
        r"\cmidrule(lr){2-4} \cmidrule(lr){6-8}",
        r"Counter & CoV & DT & MLP & & CoV & DT & MLP \\",
        r"\midrule",
    ]
    for c in COUNTERS:
        s = stats[c]
        lines.append(f"\\texttt{{{TEX[c]}}} & {cell(s['cov0'])} & {cell(s['dt0'])} & {cell(s['mlp0'])} "
                     f"& & {cell(s['cov16'])} & {cell(s['dt16'])} & {cell(s['mlp16'])} \\\\")
        if c == "ref_cycles":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    open(OUT, "w").write("\n".join(lines))
    print(f"Wrote {OUT}\n")

    # prose numbers
    print(f"{'counter':22s} {'CoV(P/E)':>12s} | {'DT P/E':>12s} | {'MLP P/E':>12s}")
    for c in COUNTERS:
        s = stats[c]
        print(f"{c:22s} {s['cov0']:5.2f}/{s['cov16']:5.2f} | "
              f"{s['dt0']:5.1f}/{s['dt16']:5.1f} | {s['mlp0']:5.1f}/{s['mlp16']:5.1f}")


if __name__ == "__main__":
    main()

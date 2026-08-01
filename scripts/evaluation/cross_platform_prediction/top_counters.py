#!/usr/bin/env python3
"""
top_counters.py
===============
Aggregate feature importances from cross_freq models (full feature set) and
report the top counters per processor × suite, averaged across all direction
pairs (e.g. 1GHz→2GHz, 1GHz→3GHz, …).

Only sample_index (a non-counter artifact) is excluded from the ranking.
Cycle counters compete like any other: `instructions`, `cpu_cycles`, and
`ref_cycles` can and do land in the top-N, because the source-side value of
the predicted quantity is genuinely the strongest single feature.

That makes this the "best K counters to deploy" ranking. It is NOT the same
question as "which non-trivial counters carry signal", which is what the
paper's counter-importance table reports -- that one additionally drops the
always-collected cycle/instruction events. Use --exclude in summary mode to
reproduce the paper's view.

Original usage (results_dir mode, called by data collection scripts):
  python top_counters.py --results_dir path/to/full --top_k 4

Analysis usage (cross_freq summary mode):
  python top_counters.py --mode summary
  python top_counters.py --mode summary --top 6 --gran 100M
"""

import argparse
import glob
import os
import sys

import pandas as pd

# ---------------------------------------------------------------------------
# Platform → human label mapping
# ---------------------------------------------------------------------------
PLATFORM_LABEL = {
    "arm_desktop": "eMAG",
    "arm_edge":    "OoO (RB5)",
    "arm_server":  "N1",
    "x86":         "x86",
    "x86_server":  "Xeon",
}

CPU_LABEL = {
    "cpu0":  "P-Core",
    "cpu4":  "OoO",
    "cpu16": "E-Core",
}

# No forced baselines (shared_features.ALWAYS_KEEP_COUNTERS == []). Every real
# counter -- including instructions/cpu_cycles/ref_cycles -- competes in the
# ranking; --top_k returns exactly top_k counters. Only sample_index (a
# non-counter artifact) is excluded from ranking.
BASELINE_COUNTERS = []
EXCLUDE = {"sample_index"}


# ---------------------------------------------------------------------------
# Original single-directory interface (used by data collection scripts)
# ---------------------------------------------------------------------------

def get_top_counters(results_dir, top_k):
    """Return the top ranked counters to pass as --input_counters so the trained
    model ends up with EXACTLY top_k counters. With no forced baselines
    (BASELINE_COUNTERS == []) this is simply the top_k most important counters."""
    imp_files = []
    for root, _dirs, files in os.walk(results_dir):
        for f in files:
            if f == "feature_importance.csv":
                imp_files.append(os.path.join(root, f))

    if not imp_files:
        return []

    n_extra = max(0, top_k - len(BASELINE_COUNTERS))

    dfs = [pd.read_csv(f) for f in imp_files]
    combined = pd.concat(dfs, ignore_index=True)
    avg = combined.groupby("feature")["importance"].mean().sort_values(ascending=False)
    avg = avg[~avg.index.isin(EXCLUDE)]
    return list(avg.head(n_extra).index)


# ---------------------------------------------------------------------------
# Summary mode: aggregate across all platforms / suites
# ---------------------------------------------------------------------------

def discover(results_root, gran, feature_set="full"):
    entries = []
    base = os.path.join(results_root, "cross_freq")
    for fi in sorted(glob.glob(
        os.path.join(base, "**", "feature_importance.csv"), recursive=True
    )):
        rel   = os.path.relpath(fi, base)
        parts = rel.split(os.sep)
        if not parts[0].endswith(f"_{gran}"):
            continue
        platform_gran = parts[0]
        platform      = "_".join(platform_gran.split("_")[:-1])

        idx = 1
        cpu = None
        if parts[idx].startswith("cpu"):
            cpu = parts[idx]
            idx += 1

        if idx + 3 > len(parts):
            continue
        suite     = parts[idx]
        fs        = parts[idx + 1]

        if fs != feature_set:
            continue

        entries.append({
            "platform": platform,
            "cpu":      cpu,
            "suite":    suite,
            "fi_path":  fi,
        })
    return entries


def human_label(platform, cpu):
    if platform == "x86" and cpu in CPU_LABEL:
        return CPU_LABEL[cpu]
    if platform == "arm_edge" and cpu in CPU_LABEL:
        return CPU_LABEL[cpu]
    return PLATFORM_LABEL.get(platform, platform)


def load_and_aggregate(entries, exclude=EXCLUDE, top_n=6):
    acc = {}
    for e in entries:
        label = human_label(e["platform"], e["cpu"])
        key   = (label, e["suite"])
        acc.setdefault(key, {})
        df = pd.read_csv(e["fi_path"])
        for _, row in df.iterrows():
            feat = row["feature"]
            imp  = float(row["importance"])
            acc[key].setdefault(feat, []).append(imp)

    rows = []
    for (label, suite), feat_dict in sorted(acc.items()):
        avg   = {f: sum(v) / len(v) for f, v in feat_dict.items()}
        total = sum(avg.values())
        pct   = {f: 100 * v / total for f, v in avg.items()}

        ranked = sorted(
            [(f, p) for f, p in pct.items() if f not in exclude],
            key=lambda x: -x[1],
        )

        for rank, (feat, imp_pct) in enumerate(ranked[:top_n], start=1):
            rows.append({
                "processor": label,
                "suite":     suite,
                "rank":      rank,
                "counter":   feat,
                "pct":       round(imp_pct, 1),
            })

    return pd.DataFrame(rows)


def pivot_table(df, top_n):
    processors = sorted(df["processor"].unique())
    for suite in sorted(df["suite"].unique()):
        sub = df[df["suite"] == suite]
        rows = []
        for rank in range(1, top_n + 1):
            row = {"Rank": rank}
            for proc in processors:
                cell = sub[(sub["processor"] == proc) & (sub["rank"] == rank)]
                if cell.empty:
                    row[proc] = "—"
                else:
                    r = cell.iloc[0]
                    row[proc] = f"{r['counter']}  {r['pct']:.1f}%"
            rows.append(row)
        tbl = pd.DataFrame(rows).set_index("Rank")
        print(f"\n{'='*80}")
        print(f"Suite: {suite}")
        print("="*80)
        print(tbl.to_string())


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode",          default="legacy",
                   choices=["legacy", "summary"],
                   help="'legacy' = original single-dir mode; 'summary' = cross-platform table")
    # Legacy mode args
    p.add_argument("--results_dir",
                   help="(legacy) Path to the 'full' directory with feature_importance.csv files.")
    p.add_argument("--top_k",        type=int,
                   help="(legacy) TOTAL feature budget (baselines + ranked counters). "
                        "Returns top_k - len(baselines) ranked counters.")
    # Summary mode args
    p.add_argument("--gran",         default="10M")
    p.add_argument("--results_root", default="results/cross_platform")
    p.add_argument("--feature_set",  default="full")
    p.add_argument("--top",          type=int, default=6)
    p.add_argument("--exclude",      nargs="*", default=list(EXCLUDE))
    args = p.parse_args()

    if args.mode == "legacy":
        if not args.results_dir or not args.top_k:
            p.error("--results_dir and --top_k are required in legacy mode")
        counters = get_top_counters(args.results_dir, args.top_k)
        if not counters:
            print(f"ERROR: No feature_importance.csv found in {args.results_dir}",
                  file=sys.stderr)
            sys.exit(1)
        print(" ".join(counters))
        return

    # Summary mode
    exclude = set(args.exclude)
    print(f"Excluding from ranking: {sorted(exclude)}")
    entries = discover(args.results_root, args.gran, args.feature_set)
    print(f"Found {len(entries)} feature_importance.csv files ({args.gran}, {args.feature_set})")
    if not entries:
        print("No entries found. Check --results_root and --gran.")
        return

    df = load_and_aggregate(entries, exclude=exclude, top_n=args.top)
    pivot_table(df, args.top)

    print("\n\nFull ranked table:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

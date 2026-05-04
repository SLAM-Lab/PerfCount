"""
multi_freq_correlation.py
=========================
Generic hardware-counter cross-correlation analyzer.

Works with any platform whose processed data follows the naming convention:
    aligned_<bench>_<freq>GHz[_cpu<N>]_phase<P>.csv

Numeric columns are auto-detected from the CSV files, so no platform-specific
counter list is required.

Examples
--------
# ARM server (single core type, all CPUs lumped together)
python multi_freq_correlation.py \\
    --data_dir ../../../../processed_data/arm_server \\
    --out_dir  ../../../../results/utils/correlation/arm_server

# x86 desktop (P-cores vs E-cores analyzed separately)
python multi_freq_correlation.py \\
    --data_dir  ../../../../processed_data/x86_desktop_heterogeneous \\
    --out_dir   ../../../../results/utils/correlation/x86_desktop_heterogeneous \\
    --cpu_ids   0 16 \\
    --cpu_names P-Core E-Core
"""

import os
import re
import glob
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


# =============================================================================
# Helpers
# =============================================================================

def discover_freqs(data_dir, cpu_id=None):
    """Return sorted list of frequency strings found in data_dir."""
    pattern = re.compile(r"aligned_.+?_([\d.]+)GHz")
    freqs = set()
    for fpath in glob.glob(os.path.join(data_dir, "aligned_*.csv")):
        fname = os.path.basename(fpath)
        if cpu_id is not None and f"_cpu{cpu_id}_" not in fname:
            continue
        m = pattern.search(fname)
        if m:
            freqs.add(m.group(1))
    return sorted(freqs, key=float)


def glob_files(data_dir, freq, cpu_id=None):
    """Return CSV paths for a given frequency (and optionally a cpu ID)."""
    if cpu_id is not None:
        pattern = os.path.join(data_dir, f"aligned_*_{freq}GHz_cpu{cpu_id}_phase*.csv")
    else:
        pattern = os.path.join(data_dir, f"aligned_*_{freq}GHz*phase*.csv")
    return glob.glob(pattern)


def load_data(files, sample_every=10):
    """Load and downsample CSVs; keep only numeric columns."""
    df_list = []
    for fpath in files:
        try:
            df = pd.read_csv(fpath)
            df = df.select_dtypes(include="number")
            df = df.drop(columns=["sample_index"], errors="ignore")
            df = df.iloc[::sample_every]
            if not df.empty:
                df_list.append(df)
        except Exception as e:
            print(f"  [WARN] Skipping {os.path.basename(fpath)}: {e}")
    if not df_list:
        return pd.DataFrame()
    # Intersect columns so concat is clean across files with different counter sets
    common_cols = df_list[0].columns
    for d in df_list[1:]:
        common_cols = common_cols.intersection(d.columns)
    return pd.concat([d[common_cols] for d in df_list], ignore_index=True)


def compute_and_save(global_df, label, freq, out_dir, threshold, corr_method):
    """Compute correlation matrix, save heatmap and redundant-pair CSV."""
    corr_matrix = global_df.corr(method=corr_method)

    # --- heatmap ---
    plt.figure(figsize=(20, 16))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(
        corr_matrix, mask=mask, cmap=cmap,
        vmax=1.0, vmin=-1.0, center=0,
        square=True, linewidths=0.5,
        cbar_kws={"shrink": 0.7}, annot=False,
    )
    plt.title(f"{label} Hardware Counter Correlation Heatmap ({freq} GHz)", fontsize=20, pad=20)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()

    slug = label.lower().replace(" ", "_").replace("-", "")
    heatmap_path = os.path.join(out_dir, f"counter_correlation_heatmap_{slug}_{freq}GHz.png")
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    print(f"  [+] Saved heatmap → {heatmap_path}")

    # --- redundant pairs ---
    redundant = []
    cols = corr_matrix.columns.tolist()
    for i in range(len(cols)):
        for j in range(i):
            val = corr_matrix.iloc[i, j]
            if abs(val) > threshold:
                redundant.append({"Counter 1": cols[i], "Counter 2": cols[j], "Correlation": val})

    redundant_df = pd.DataFrame(redundant)
    if not redundant_df.empty:
        redundant_df = redundant_df.sort_values("Correlation", key=abs, ascending=False)

    csv_path = os.path.join(out_dir, f"redundant_counters_{slug}_{freq}GHz.csv")
    redundant_df.to_csv(csv_path, index=False)
    print(f"  [+] {len(redundant_df)} highly correlated pairs (>{threshold}) → {csv_path}")

    if not redundant_df.empty:
        print(f"\n  --- Top pairs for {label} @ {freq} GHz ---")
        print(redundant_df.head(5).to_string(index=False))


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Cross-correlation analysis for hardware performance counters."
    )
    parser.add_argument("--data_dir", required=True,
                        help="Directory containing aligned_*.csv files")
    parser.add_argument("--out_dir", required=True,
                        help="Output directory for heatmaps and CSVs")
    parser.add_argument("--freqs", nargs="*", default=None,
                        help="Frequencies to analyze (e.g. 1.0 2.0 3.0). "
                             "Auto-detected from filenames if omitted.")
    parser.add_argument("--cpu_ids", nargs="*", default=None,
                        help="CPU IDs to analyze separately (e.g. 0 16). "
                             "Omit to lump all CPUs together.")
    parser.add_argument("--cpu_names", nargs="*", default=None,
                        help="Display names for each cpu_id (e.g. P-Core E-Core). "
                             "Defaults to 'cpu<N>' if omitted.")
    parser.add_argument("--threshold", type=float, default=0.90,
                        help="Correlation magnitude threshold for flagging redundant pairs (default: 0.90)")
    parser.add_argument("--corr_method", choices=["pearson", "spearman", "kendall"],
                        default="spearman",
                        help="Correlation method (default: spearman)")
    parser.add_argument("--sample_every", type=int, default=10,
                        help="Keep every Nth row from each file (default: 10)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Build CPU groups: list of (cpu_id_or_None, display_name)
    if args.cpu_ids:
        names = args.cpu_names if args.cpu_names else [f"cpu{c}" for c in args.cpu_ids]
        if len(names) != len(args.cpu_ids):
            parser.error("--cpu_names must have the same number of entries as --cpu_ids")
        cpu_groups = list(zip(args.cpu_ids, names))
    else:
        cpu_groups = [(None, "all")]

    print(f"\n{'='*60}")
    print(f"  data_dir     : {os.path.abspath(args.data_dir)}")
    print(f"  out_dir      : {os.path.abspath(args.out_dir)}")
    print(f"  corr_method  : {args.corr_method}  threshold={args.threshold}")
    print(f"  cpu_groups   : {cpu_groups}")
    print(f"{'='*60}\n")

    for cpu_id, cpu_name in cpu_groups:
        freqs = args.freqs if args.freqs else discover_freqs(args.data_dir, cpu_id)
        if not freqs:
            print(f"[WARN] No frequencies found for cpu_id={cpu_id} in {args.data_dir}")
            continue

        for freq in freqs:
            print(f"\n{'='*60}")
            print(f"  {cpu_name}  |  {freq} GHz")
            print(f"{'='*60}")

            files = glob_files(args.data_dir, freq, cpu_id)
            if not files:
                print(f"  [-] No files found. Skipping.")
                continue
            print(f"  [*] Found {len(files)} files. Loading...")

            global_df = load_data(files, sample_every=args.sample_every)
            if global_df.empty:
                print(f"  [-] No usable data after loading. Skipping.")
                continue
            print(f"  [*] Dataset: {len(global_df)} samples x {len(global_df.columns)} counters. "
                  f"Computing {args.corr_method} correlation...")

            compute_and_save(
                global_df, cpu_name, freq,
                args.out_dir, args.threshold, args.corr_method,
            )


if __name__ == "__main__":
    main()

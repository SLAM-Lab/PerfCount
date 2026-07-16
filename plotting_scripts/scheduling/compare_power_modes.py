import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def plot_pct_diff_by_policy(df, metric, out_file):
    """Bar chart of mean Pct_Diff per policy, centered on 0%."""
    plot_df = df[df['Metric'] == metric].sort_values(by='Mean_Pct_Diff')
    if plot_df.empty:
        return

    policies = plot_df['Policy'].tolist()
    values = plot_df['Mean_Pct_Diff'].tolist()
    colors = ['tab:red' if v < 0 else 'tab:blue' for v in values]

    plt.figure(figsize=(12, 7))
    display_keys = [p.replace('_', '\n') for p in policies]
    plt.bar(display_keys, values, color=colors, edgecolor='black')
    plt.axhline(0.0, color='black', linestyle='--', linewidth=1)
    plt.title(f"Per-Sample vs Baseline Power: Mean % Change in Final {metric}")
    plt.ylabel(f"% change in Final {metric}\n(per-sample power vs baseline power)")
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_file, dpi=100)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Compare scheduling-simulator results between per-sample and baseline power models."
    )
    parser.add_argument('--persample_summary', type=str, required=True,
                         help="Path to all_phases_summary.csv from the --power_mode per_sample run")
    parser.add_argument('--baseline_summary', type=str, required=True,
                         help="Path to all_phases_summary.csv from the --power_mode baseline run")
    parser.add_argument('--out_dir', type=str, required=True)
    args = parser.parse_args()

    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    persample_df = pd.read_csv(args.persample_summary)
    baseline_df = pd.read_csv(args.baseline_summary)

    keys = ['Workload', 'Phase', 'Metric', 'Policy']
    merged = persample_df.merge(baseline_df, on=keys, suffixes=('_PerSample', '_Baseline'))

    merged['Abs_Diff'] = merged['Final_Value_PerSample'] - merged['Final_Value_Baseline']
    merged['Pct_Diff'] = 100 * merged['Abs_Diff'] / merged['Final_Value_Baseline']

    diff_cols = keys + ['Final_Value_PerSample', 'Final_Value_Baseline', 'Abs_Diff', 'Pct_Diff']
    merged[diff_cols].to_csv(out_path / "power_mode_diff.csv", index=False)

    by_policy = merged.groupby(['Metric', 'Policy'])['Pct_Diff'].agg(
        Mean_Pct_Diff='mean', Median_Pct_Diff='median', Std_Pct_Diff='std'
    ).reset_index()
    by_policy = by_policy.sort_values(by=['Metric', 'Mean_Pct_Diff'])
    by_policy.to_csv(out_path / "power_mode_diff_by_policy.csv", index=False)

    for metric in by_policy['Metric'].unique():
        plot_pct_diff_by_policy(by_policy, metric, out_path / f"power_mode_diff_{metric}.png")

    print(f"Compared {len(merged)} rows across {merged['Workload'].nunique()} workloads.")
    print(f"Wrote diff CSVs and plots to {out_path}")


if __name__ == "__main__":
    main()

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

BASELINE_POLICY = 'Proactive_Hetero_Oracle'


def normalize(df):
    """Normalize Final_Value within each (Workload, Phase, Metric) group so that
    BASELINE_POLICY == 1.0, then average the normalized score per (Metric, Policy)."""
    ref = df[df['Policy'] == BASELINE_POLICY][['Workload', 'Phase', 'Metric', 'Final_Value']]
    ref = ref.rename(columns={'Final_Value': 'Ref_Value'})
    merged = df.merge(ref, on=['Workload', 'Phase', 'Metric'])
    merged['Normalized'] = merged['Final_Value'] / merged['Ref_Value']
    return merged.groupby(['Metric', 'Policy'])['Normalized'].mean().reset_index()


def plot_grouped(metric_df, metric, out_file, y_max=3.0):
    metric_df = metric_df.sort_values('PerSample')
    policies = metric_df['Policy'].tolist()
    spacing = 2.0
    x = [i * spacing for i in range(len(policies))]
    width = 0.7

    plt.figure(figsize=(max(24, len(policies) * spacing * 0.4), 8))
    bars_a = plt.bar([i - width / 2 for i in x], metric_df['PerSample'], width, label='Per-Sample Power', edgecolor='black')
    bars_b = plt.bar([i + width / 2 for i in x], metric_df['Baseline'], width, label='Baseline Power', edgecolor='black')

    for bars, vals in ((bars_a, metric_df['PerSample']), (bars_b, metric_df['Baseline'])):
        for bar, val in zip(bars, vals):
            if val > y_max:
                plt.text(bar.get_x() + bar.get_width() / 2.0, y_max - 0.05, f"{val:.1f}",
                         ha='center', va='top', rotation=90, color='red', fontsize=8,
                         fontweight='bold', bbox=dict(facecolor='white', alpha=0.9, edgecolor='red', pad=1))

    plt.axhline(1.0, color='red', linestyle='--', linewidth=1.5, label=f'1.0 = {BASELINE_POLICY}')
    plt.ylim(0, y_max)
    plt.xticks(x, [p.replace('_', '\n') for p in policies], rotation=45, ha='right', fontsize=9)
    plt.ylabel(f"Relative {metric}\n(normalized to {BASELINE_POLICY} within each power model)")
    plt.title(f"Policy Comparison: Per-Sample vs Baseline Power ({metric})")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_file, dpi=100)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Grouped bar chart comparing policies under per-sample vs baseline power, "
                     "each normalized within its own power model so Proactive_Hetero_Oracle == 1.0."
    )
    parser.add_argument('--persample_summary', type=str, required=True)
    parser.add_argument('--baseline_summary', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    args = parser.parse_args()

    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    persample_df = pd.read_csv(args.persample_summary)
    baseline_df = pd.read_csv(args.baseline_summary)

    persample_norm = normalize(persample_df).rename(columns={'Normalized': 'PerSample'})
    baseline_norm = normalize(baseline_df).rename(columns={'Normalized': 'Baseline'})

    merged = persample_norm.merge(baseline_norm, on=['Metric', 'Policy'])
    merged.to_csv(out_path / "policy_comparison_normalized.csv", index=False)

    for metric in merged['Metric'].unique():
        metric_df = merged[merged['Metric'] == metric]
        out_file = out_path / f"policy_comparison_normalized_{metric}.png"
        plot_grouped(metric_df, metric, out_file)
        print(f"Wrote {out_file}")


if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

TEMPORAL_AXES = ['No Temporal', 'Workload Forecasting', 'Proactive Oracle']
DECISION_AXES = ['Cheap/SOTA', 'Model', 'Oracle']

TAXONOMY = {
    ('No Temporal', 'Cheap/SOTA'): [
        'Performance_Gov_P', 'Ondemand_P', 'Ondemand_E',
        'Conservative_P', 'Conservative_E', 'Random_P', 'Random_E',
        'EAS_Hetero', 'EAS_With_DVFS', 'Threshold_Migration',
        'Thread_Director', 'Micro_EAS',
    ],
    ('No Temporal', 'Model'): [
        'Reactive_1_Step_P', 'Reactive_1_Step_E',
        'Reactive_Combined_W1', 'Reactive_Combined_W5', 'Reactive_Combined_W10',
        'UCB1_P', 'UCB1_E', 'UCB1_Hetero',
    ],
    ('No Temporal', 'Oracle'): [
        'Greedy_Oracle_P', 'Greedy_Oracle_E',
    ],
    ('Workload Forecasting', 'Cheap/SOTA'): [
        'Schedutil_PELT_P', 'Schedutil_PELT_E', 'EWMA_P', 'EWMA_E', 'Intel_HWP_P',
    ],
    ('Workload Forecasting', 'Model'): [],
    ('Workload Forecasting', 'Oracle'): [],
    ('Proactive Oracle', 'Cheap/SOTA'): [],
    ('Proactive Oracle', 'Model'): [
        'MPC_Oracle_P_W5', 'MPC_Oracle_P_W10', 'MPC_Oracle_E_W5', 'MPC_Oracle_E_W10',
        'MPC_Oracle_Combined_W1', 'MPC_Oracle_Combined_W5', 'MPC_Oracle_Combined_W10',
    ],
    ('Proactive Oracle', 'Oracle'): [
        'Global_Oracle_P', 'Global_Oracle_E', 'Proactive_Hetero_Oracle',
        'IsoFreq_Oracle_1.0GHz', 'IsoFreq_Oracle_2.0GHz',
        'IsoFreq_Oracle_3.0GHz', 'IsoFreq_Oracle_4.0GHz',
    ],
}
BASELINE_POLICY = 'Proactive_Hetero_Oracle'


def plot_grid(grid_df, metric, out_file):
    values = np.full((len(TEMPORAL_AXES), len(DECISION_AXES)), np.nan)
    counts = np.full((len(TEMPORAL_AXES), len(DECISION_AXES)), 0)

    for _, row in grid_df.iterrows():
        i = TEMPORAL_AXES.index(row['Temporal'])
        j = DECISION_AXES.index(row['Decision'])
        values[i, j] = row['Mean_Normalized']
        counts[i, j] = row['N']

    fig, ax = plt.subplots(figsize=(10, 7))
    masked = np.ma.masked_invalid(values)
    cmap = plt.get_cmap('RdYlGn_r')
    im = ax.imshow(masked, cmap=cmap, vmin=1.0, vmax=np.nanmax(values) if np.isfinite(np.nanmax(values)) else 2.0)

    for i in range(len(TEMPORAL_AXES)):
        for j in range(len(DECISION_AXES)):
            if np.isnan(values[i, j]):
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor='lightgray', edgecolor='black'))
                ax.text(j, i, "N/A", ha='center', va='center', fontsize=12, color='black')
            else:
                ax.text(j, i, f"{values[i, j]:.2f}x\n(n={counts[i, j]})",
                        ha='center', va='center', fontsize=12, fontweight='bold', color='black')

    ax.set_xticks(range(len(DECISION_AXES)))
    ax.set_xticklabels(DECISION_AXES, fontsize=11)
    ax.set_yticks(range(len(TEMPORAL_AXES)))
    ax.set_yticklabels(TEMPORAL_AXES, fontsize=11)
    ax.set_xlabel("Decision Making")
    ax.set_ylabel("Temporal Prediction")
    ax.set_title(f"Policy Taxonomy: Temporal Prediction x Decision Making ({metric})\n"
                  f"(Mean Final_Value normalized to {BASELINE_POLICY} == 1.0)")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(f"Relative {metric} (lower = better)")

    plt.tight_layout()
    plt.savefig(out_file, dpi=100)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="3x3 grid (Temporal Prediction x Decision Making) of mean normalized "
                     "EDP/ED2P relative to Proactive_Hetero_Oracle."
    )
    parser.add_argument('--summary_csv', type=str, required=True,
                         help="Path to all_phases_summary.csv")
    parser.add_argument('--out_dir', type=str, required=True)
    args = parser.parse_args()

    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.summary_csv)

    base = df[df['Policy'] == BASELINE_POLICY][['Workload', 'Phase', 'Metric', 'Final_Value']]
    base = base.rename(columns={'Final_Value': 'Base_Value'})
    merged = df.merge(base, on=['Workload', 'Phase', 'Metric'])
    merged['Normalized'] = merged['Final_Value'] / merged['Base_Value']

    rows = []
    for (temporal, decision), policies in TAXONOMY.items():
        if not policies:
            continue
        for metric in ['EDP', 'ED2P']:
            subset = merged[(merged['Metric'] == metric) & (merged['Policy'].isin(policies))]
            if subset.empty:
                continue
            rows.append({
                'Temporal': temporal,
                'Decision': decision,
                'Metric': metric,
                'Mean_Normalized': subset['Normalized'].mean(),
                'N': len(subset),
            })

    grid_df = pd.DataFrame(rows)
    grid_df.to_csv(out_path / "policy_taxonomy.csv", index=False)
    print(grid_df.to_string(index=False))

    for metric in ['EDP', 'ED2P']:
        metric_df = grid_df[grid_df['Metric'] == metric]
        out_file = out_path / f"policy_taxonomy_{metric}.png"
        plot_grid(metric_df, metric, out_file)
        print(f"Wrote {out_file}")


if __name__ == "__main__":
    main()

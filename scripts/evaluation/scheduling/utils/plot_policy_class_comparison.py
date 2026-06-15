import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# Domain: which "axis" of the problem a policy operates on.
DVFS_POLICIES = {
    'Proactive': [
        'Greedy_Oracle_P', 'MPC_Oracle_P_W5', 'MPC_Oracle_P_W10', 'Global_Oracle_P',
        'Greedy_Oracle_E', 'MPC_Oracle_E_W5', 'MPC_Oracle_E_W10', 'Global_Oracle_E',
    ],
    'Reactive': [
        'Reactive_1_Step_P', 'Reactive_1_Step_E',
    ],
    'SOTA': [
        'Performance_Gov_P', 'Ondemand_P', 'Conservative_P', 'Schedutil_PELT_P', 'Intel_HWP_P',
        'EWMA_P', 'UCB1_P', 'Random_P',
        'Ondemand_E', 'Conservative_E', 'Schedutil_PELT_E', 'EWMA_E', 'UCB1_E', 'Random_E',
    ],
}

# Combined core-placement + per-core DVFS policies.
SCHEDULING_POLICIES = {
    'Proactive': [
        'Proactive_Hetero_Oracle',
        'MPC_Oracle_Combined_W1', 'MPC_Oracle_Combined_W5', 'MPC_Oracle_Combined_W10',
    ],
    'Reactive': [
        'Reactive_Combined_W1', 'Reactive_Combined_W5', 'Reactive_Combined_W10',
    ],
    'SOTA': [
        'EAS_Hetero', 'EAS_With_DVFS', 'Threshold_Migration', 'Thread_Director',
        'Micro_EAS', 'UCB1_Hetero',
    ],
}

# Pure core-placement ablations (oracle scheduling at a fixed frequency).
# No reactive/SOTA equivalents exist in isolation, so those classes are empty.
SCHEDULING_ISOFREQ_POLICIES = {
    'Proactive': [
        'IsoFreq_Oracle_1.0GHz', 'IsoFreq_Oracle_2.0GHz', 'IsoFreq_Oracle_3.0GHz', 'IsoFreq_Oracle_4.0GHz',
    ],
    'Reactive': [],
    'SOTA': [],
}

DOMAINS = {
    'DVFS': DVFS_POLICIES,
    'Scheduling': SCHEDULING_POLICIES,
    'Scheduling-IsoFreq': SCHEDULING_ISOFREQ_POLICIES,
}
CLASSES = ['Proactive', 'Reactive', 'SOTA']
BASELINE_POLICY = 'Proactive_Hetero_Oracle'


def main():
    parser = argparse.ArgumentParser(
        description="Bar plot of overall average EDP vs ED2P (EDDP), normalized to the global "
                     "Proactive_Hetero_Oracle, grouped by Proactive/Reactive/SOTA policy class "
                     "within the DVFS and Scheduling domains."
    )
    parser.add_argument('--summary_csv', type=str, required=True,
                         help="Path to all_phases_summary.csv")
    parser.add_argument('--out_dir', type=str, required=True)
    args = parser.parse_args()

    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.summary_csv)

    # Normalize each (Workload, Phase, Metric) row to the global oracle for that combo.
    base = df[df['Policy'] == BASELINE_POLICY][['Workload', 'Phase', 'Metric', 'Final_Value']]
    base = base.rename(columns={'Final_Value': 'Base_Value'})
    merged = df.merge(base, on=['Workload', 'Phase', 'Metric'])
    merged['Normalized'] = merged['Final_Value'] / merged['Base_Value']

    rows = []
    for domain, classes in DOMAINS.items():
        for cls, policies in classes.items():
            if not policies:
                continue
            for metric in ['EDP', 'ED2P']:
                subset = merged[(merged['Metric'] == metric) & (merged['Policy'].isin(policies))]
                if subset.empty:
                    continue
                rows.append({
                    'Domain': domain,
                    'Class': cls,
                    'Metric': metric,
                    'Mean_Normalized': subset['Normalized'].mean(),
                    'N': len(subset),
                })

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(out_path / "policy_class_comparison.csv", index=False)
    print(summary_df.to_string(index=False))

    # --- Grouped bar plot: x = Domain x Class, side-by-side bars = EDP / ED2P ---
    fig, ax = plt.subplots(figsize=(10, 6))

    group_keys = [(d, c) for d in DOMAINS for c in CLASSES if DOMAINS[d][c]]
    group_labels = [f"{d}\n{c}" for d, c in group_keys]
    x = range(len(group_labels))
    width = 0.35

    edp_vals = []
    ed2p_vals = []
    for d, c in group_keys:
        row = summary_df[(summary_df['Domain'] == d) & (summary_df['Class'] == c)]
        edp = row[row['Metric'] == 'EDP']['Mean_Normalized']
        ed2p = row[row['Metric'] == 'ED2P']['Mean_Normalized']
        edp_vals.append(edp.values[0] if len(edp) else float('nan'))
        ed2p_vals.append(ed2p.values[0] if len(ed2p) else float('nan'))

    ax.bar([i - width / 2 for i in x], edp_vals, width, label='EDP', color='tab:blue', edgecolor='black')
    ax.bar([i + width / 2 for i in x], ed2p_vals, width, label='ED2P (EDDP)', color='tab:orange', edgecolor='black')

    ax.axhline(1.0, color='red', linestyle='--', linewidth=1.5, label=f'1.0 = {BASELINE_POLICY}')
    ax.set_xticks(list(x))
    ax.set_xticklabels(group_labels)
    ax.set_ylabel("Mean Normalized Score\n(relative to Global Oracle, lower = better)")
    ax.set_title("Overall Average EDP vs ED2P (EDDP)\nProactive vs Reactive vs SOTA, DVFS vs Scheduling")
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    out_file = out_path / "policy_class_comparison.png"
    plt.savefig(out_file, dpi=150)
    plt.close()

    print(f"\nSaved {out_path / 'policy_class_comparison.csv'} and {out_file}")


if __name__ == "__main__":
    main()

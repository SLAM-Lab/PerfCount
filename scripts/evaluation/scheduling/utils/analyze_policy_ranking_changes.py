import pandas as pd
from pathlib import Path
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Check whether the best (lowest EDP/ED2P) policy per workload/phase "
                     "differs between the per-sample and baseline power runs."
    )
    parser.add_argument('--persample_summary', type=str, required=True)
    parser.add_argument('--baseline_summary', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    args = parser.parse_args()

    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    persample_df = pd.read_csv(args.persample_summary)
    baseline_df = pd.read_csv(args.baseline_summary)

    def best_policy(df):
        idx = df.groupby(['Workload', 'Phase', 'Metric'])['Final_Value'].idxmin()
        return df.loc[idx, ['Workload', 'Phase', 'Metric', 'Policy', 'Final_Value']].reset_index(drop=True)

    best_persample = best_policy(persample_df)
    best_baseline = best_policy(baseline_df)

    keys = ['Workload', 'Phase', 'Metric']
    merged = best_persample.merge(best_baseline, on=keys, suffixes=('_PerSample', '_Baseline'))
    merged['Changed'] = merged['Policy_PerSample'] != merged['Policy_Baseline']

    merged.to_csv(out_path / "policy_ranking_changes.csv", index=False)

    by_metric = merged.groupby('Metric')['Changed'].agg(['sum', 'count', 'mean']).rename(
        columns={'sum': 'N_Changed', 'count': 'N_Total', 'mean': 'Frac_Changed'}
    ).reset_index()
    by_metric.to_csv(out_path / "policy_ranking_changes_by_metric.csv", index=False)

    print(by_metric.to_string(index=False))
    print(f"\n{merged['Changed'].sum()} / {len(merged)} workload/phase/metric combos picked a different best policy.")
    print(f"Wrote {out_path / 'policy_ranking_changes.csv'} and {out_path / 'policy_ranking_changes_by_metric.csv'}")


if __name__ == "__main__":
    main()

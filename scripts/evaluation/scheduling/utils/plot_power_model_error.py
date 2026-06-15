import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Plot % difference between measured per-sample power and the fixed "
                     "baseline lookup power, relative to each config's own baseline value."
    )
    parser.add_argument('--error_csv', type=str, required=True,
                         help="Path to power_model_error.csv")
    parser.add_argument('--out_dir', type=str, required=True)
    args = parser.parse_args()

    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.error_csv).sort_values('Config')

    colors = ['tab:blue' if c.startswith('P_') else 'tab:orange' for c in df['Config']]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(df['Config'], df['Pct_Bias'], color=colors, edgecolor='black')
    for bar, val in zip(bars, df['Pct_Bias']):
        plt.text(bar.get_x() + bar.get_width() / 2.0, val + 1, f"{val:.1f}%",
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.axhline(0.0, color='black', linestyle='--', linewidth=1)
    plt.title("Measured Per-Sample Power vs. Baseline Lookup Power\n(% difference, relative to each config's own baseline value)")
    plt.ylabel("% difference (measured - baseline) / baseline")
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    out_file = out_path / "power_model_pct_bias.png"
    plt.savefig(out_file, dpi=100)
    plt.close()

    print(f"Wrote {out_file}")


if __name__ == "__main__":
    main()

import sys
import re
import pandas as pd
import numpy as np
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from data_loader import get_power_w


def main():
    parser = argparse.ArgumentParser(
        description="Quantify how far the fixed get_power_w() lookup table deviates "
                     "from measured per-sample power (RAPL) across all workloads."
    )
    parser.add_argument('--input_dir', type=str, required=True,
                         help="Directory containing speedups_*.csv granular phase traces")
    parser.add_argument('--out_dir', type=str, required=True)
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    pattern = re.compile(r"speedups_([PE]_[0-9.]+GHz)_(.+)_phase(\d+)\.csv")
    pairs = sorted(set(m.groups()[1:] for f in input_path.glob("speedups_*.csv") if (m := pattern.search(f.name))))

    measured = {}  # cfg -> list of arrays
    for wl, ph in pairs:
        for sf in input_path.glob(f"speedups_*_{wl}_phase{ph}.csv"):
            df = pd.read_csv(sf).dropna()
            power_cols = [c for c in df.columns if c.startswith('Power_')]
            if not power_cols:
                continue
            for col in power_cols:
                cfg = col.replace('Power_', '')
                measured.setdefault(cfg, []).append(df[col].values)
            break  # only one file per workload/phase carries the Power_ columns

    rows = []
    for cfg, arrays in sorted(measured.items()):
        vals = np.concatenate(arrays)
        lookup = get_power_w(cfg)
        diff = vals - lookup
        rows.append({
            'Config': cfg,
            'Lookup_W': lookup,
            'Measured_Mean_W': vals.mean(),
            'Measured_Std_W': vals.std(),
            'N_Samples': len(vals),
            'MAE_W': np.abs(diff).mean(),
            'RMSE_W': np.sqrt((diff ** 2).mean()),
            'Mean_Bias_W': diff.mean(),
            'Pct_Bias': 100 * diff.mean() / lookup,
            'MAPE_pct': 100 * np.abs(diff / lookup).mean(),
        })

    result = pd.DataFrame(rows).sort_values('Config')
    result.to_csv(out_path / "power_model_error.csv", index=False)
    print(result.to_string(index=False))
    print(f"\nWrote {out_path / 'power_model_error.csv'}")


if __name__ == "__main__":
    main()

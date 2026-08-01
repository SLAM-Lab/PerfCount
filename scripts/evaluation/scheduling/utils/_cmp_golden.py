#!/usr/bin/env python3
"""Compare a simulator CSV against its golden copy, ignoring row order.

main.py gathers per-workload results with concurrent.futures.as_completed, so row order tracks
worker completion and is not stable across runs. Values must be identical; order need not be.
Exits 0 when they match, 1 otherwise, printing the columns that moved and the worst case.
"""
import sys
import pandas as pd

a, b = pd.read_csv(sys.argv[1]), pd.read_csv(sys.argv[2])
name = sys.argv[2].rsplit('/', 1)[-1]

if list(a.columns) != list(b.columns):
    print(f"MISMATCH: {name}\n  columns differ:\n    golden {list(a.columns)}\n    now    {list(b.columns)}")
    sys.exit(1)
if len(a) != len(b):
    print(f"MISMATCH: {name}\n  row count {len(a)} -> {len(b)}")
    sys.exit(1)

key = [c for c in ('Workload', 'Phase', 'Metric', 'Policy') if c in a.columns]
m = a.merge(b, on=key, suffixes=('_g', '_n'), how='outer', indicator=True)
if (m['_merge'] != 'both').any():
    miss = m[m['_merge'] != 'both']
    print(f"MISMATCH: {name}\n  {len(miss)} rows present in only one file, e.g.")
    print('   ', miss.iloc[0][key].to_dict())
    sys.exit(1)

bad = False
for c in a.columns:
    if c in key or f'{c}_g' not in m:
        continue
    g, n = m[f'{c}_g'], m[f'{c}_n']
    if pd.api.types.is_numeric_dtype(g) and pd.api.types.is_numeric_dtype(n):
        d = (g - n).abs()
        if d.max() > 0:
            w = m.loc[d.idxmax()]
            print(f"MISMATCH: {name}\n  {c}: {int((d > 0).sum())}/{len(m)} rows differ, "
                  f"max |diff|={d.max():.6g}")
            print(f"    worst: {' '.join(str(w[k]) for k in key)}  "
                  f"{w[f'{c}_g']:.6g} -> {w[f'{c}_n']:.6g}")
            bad = True
    elif (g.astype(str) != n.astype(str)).any():
        print(f"MISMATCH: {name}\n  {c}: {int((g.astype(str) != n.astype(str)).sum())} rows differ")
        bad = True
sys.exit(1 if bad else 0)

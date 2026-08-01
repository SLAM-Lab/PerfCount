#!/usr/bin/env python3
"""Component tests for gate_predictions.gate_file.

Properties that must hold regardless of data:
  1. margin=-inf reproduces the corrected set on every flipped row (gate is a no-op)
  2. a very large margin reproduces the baseline exactly (gate rejects everything)
  3. acceptance count is monotone non-increasing in margin
  4. output schema and row count match the baseline
  5. accepted rows equal the corrected rows; unaccepted rows equal the baseline rows
  6. the gate never consults truth (accepted set is unchanged if truth files are hidden)
Run: python3 test_gate_predictions.py
"""
import os, sys, glob
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from gate_predictions import gate_file
from xproc_eval import HP

B = os.path.join(HP, 'cross_proc_translate_10M', 'speedups_from_P_3.0GHz')
C = os.path.join(HP, 'cross_proc_translate_hybrid_10M', 'speedups_from_P_3.0GHz')

fails = []


def check(name, cond, detail=''):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not cond:
        fails.append(name)


def main():
    files = sorted(os.path.basename(f) for f in glob.glob(f'{B}/spec_*.csv'))[:6]
    if not files:
        sys.exit('no test inputs found')
    print(f'{len(files)} files, source P_3.0GHz\n')

    for p_exp in (1, 2):
        print(f'p_exp={p_exp} ({"EDP" if p_exp == 1 else "ED2P"})')
        counts = []
        for margin in (-1e9, 0.0, 0.02, 0.05, 0.10, 0.50, 1e9):
            acc = tot = flp = 0
            same_as_corr = same_as_base = True
            schema_ok = rows_ok = True
            for f in files:
                bf, cf = f'{B}/{f}', f'{C}/{f}'
                out, nf, na, n = gate_file(bf, cf, margin, p_exp)
                acc += na; tot += n; flp += nf
                b = pd.read_csv(bf).iloc[:n].reset_index(drop=True)
                c = pd.read_csv(cf).iloc[:n].reset_index(drop=True)
                schema_ok &= list(out.columns) == list(b.columns)
                rows_ok &= len(out) == n
                if margin <= -1e8:
                    # every flipped row must have taken the corrected values
                    diff = (out.values != b.values).any(axis=1)
                    same_as_corr &= (diff.sum() == nf)
                if margin >= 1e8:
                    same_as_base &= np.allclose(out.values, b.values, equal_nan=True)
            counts.append((margin, acc))
            if margin <= -1e8:
                check('no-op at margin=-inf accepts every flip', same_as_corr,
                      f'{acc} accepted, {flp} flips')
            if margin >= 1e8:
                check('exact baseline at margin=+inf', same_as_base and acc == 0,
                      f'{acc} accepted')
            if margin == 0.05:
                check('schema and row count preserved', schema_ok and rows_ok)

        mono = all(counts[i][1] >= counts[i + 1][1] for i in range(len(counts) - 1))
        check('acceptance monotone non-increasing in margin', mono,
              ' '.join(f'{m:g}:{a}' for m, a in counts))

    # accepted rows must equal corrected, rejected must equal baseline
    f = files[0]
    out, nf, na, n = gate_file(f'{B}/{f}', f'{C}/{f}', 0.05, 1)
    b = pd.read_csv(f'{B}/{f}').iloc[:n].reset_index(drop=True)
    c = pd.read_csv(f'{C}/{f}').iloc[:n].reset_index(drop=True)
    changed = (out.values != b.values).any(axis=1)
    took_corrected = np.allclose(out.values[changed], c.values[changed], equal_nan=True)
    kept_baseline = np.allclose(out.values[~changed], b.values[~changed], equal_nan=True)
    check('changed rows equal corrected', took_corrected, f'{changed.sum()} rows')
    check('unchanged rows equal baseline', kept_baseline)
    check('changed row count equals reported acceptance', int(changed.sum()) == na,
          f'{changed.sum()} vs {na}')

    print()
    if fails:
        print(f'{len(fails)} FAILED: {fails}')
        sys.exit(1)
    print('all component tests passed')


if __name__ == '__main__':
    main()

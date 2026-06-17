"""
plotter.py — all plotting and CSV export for the scheduling simulator.

Called from main() after all workloads finish, using the aggregated
all_phases_summary-style DataFrame (columns: Workload, Phase, Metric,
Policy, Final_Value).

Generates:
  Per-workload bars (phases averaged):
    bar_wl/DVFS_{metric}_{wl}.png
    bar_wl/IsoFreq_{metric}_{wl}.png
    bar_wl/Hetero_{metric}_{wl}.png

  Suite-averaged bars:
    bar_suite/DVFS_{metric}_{suite}.png
    bar_suite/IsoFreq_{metric}_{suite}.png
    bar_suite/Hetero_{metric}_{suite}.png

  Taxonomy breakdown (normalized to oracle, placeholders shown):
    taxonomy/suite_level/{cat}_{metric}_{suite}.png
    taxonomy/suite_level/IsoFreq_{freq}_{metric}_{suite}.png
    taxonomy/{suite_dir}/per_workload/{cat}_{metric}_{wl}.png
    taxonomy/{suite_dir}/per_workload/IsoFreq_{freq}_{metric}_{wl}.png

  CSVs:
    csv/per_workload_avg.csv
    csv/suite_avg_all_policies.csv
    csv/per_workload_{category}.csv
    csv/suite_avg_{category}.csv
    csv/taxonomy_{metric}_{category}_{suite}.csv
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path


# ─────────────────────────────────────────────────────────────
# Policy groups  (ordered for display)
# ─────────────────────────────────────────────────────────────

P_DVFS_POLICIES = [
    'Static_P_1.0GHz', 'Static_P_2.0GHz', 'Static_P_3.0GHz', 'Static_P_4.0GHz',
    'Performance_Gov_P', 'Ondemand_P', 'Conservative_P',
    'Schedutil_PELT_P', 'Intel_HWP_P', 'EWMA_P', 'UCB1_P',
    'Reactive_Oracle_P',
    'Model_Greedy_P', 'Model_Greedy_Oracle_P', 'Model_Global_P',
    'Ondemand_Future_P', 'Conservative_Future_P', 'Schedutil_Future_P',
    'Intel_HWP_Future_P', 'EWMA_Future_P', 'UCB1_Future_P',
    'Greedy_Oracle_P', 'Global_Oracle_P',
]

E_DVFS_POLICIES = [
    'Static_E_1.0GHz', 'Static_E_2.0GHz', 'Static_E_3.0GHz', 'Static_E_4.0GHz',
    'Performance_Gov_E', 'Ondemand_E', 'Conservative_E',
    'Schedutil_PELT_E', 'Intel_HWP_E', 'EWMA_E', 'UCB1_E',
    'Reactive_Oracle_E',
    'Model_Greedy_E', 'Model_Greedy_Oracle_E', 'Model_Global_E',
    'Ondemand_Future_E', 'Conservative_Future_E', 'Schedutil_Future_E',
    'Intel_HWP_Future_E', 'EWMA_Future_E', 'UCB1_Future_E',
    'Greedy_Oracle_E', 'Global_Oracle_E',
]

# Combined list kept for backward compatibility
DVFS_POLICIES = P_DVFS_POLICIES + E_DVFS_POLICIES

ISOFREQ_POLICIES = [
    'Static_P_1.0GHz', 'Static_E_1.0GHz',
    'Static_P_2.0GHz', 'Static_E_2.0GHz',
    'Static_P_3.0GHz', 'Static_E_3.0GHz',
    'Static_P_4.0GHz', 'Static_E_4.0GHz',
    'Micro_EAS', 'EAS_Hetero', 'UCB1_Hetero',
    'Model_IsoFreq_1.0GHz', 'Model_IsoFreq_2.0GHz',
    'Model_IsoFreq_3.0GHz', 'Model_IsoFreq_4.0GHz',
    'IsoFreq_Reactive_Oracle_1.0GHz', 'IsoFreq_Reactive_Oracle_2.0GHz',
    'IsoFreq_Reactive_Oracle_3.0GHz', 'IsoFreq_Reactive_Oracle_4.0GHz',
    'IsoFreq_Oracle_Heuristic_1.0GHz', 'IsoFreq_Oracle_Heuristic_2.0GHz',
    'IsoFreq_Oracle_Heuristic_3.0GHz', 'IsoFreq_Oracle_Heuristic_4.0GHz',
    'IsoFreq_Model_Oracle_1.0GHz', 'IsoFreq_Model_Oracle_2.0GHz',
    'IsoFreq_Model_Oracle_3.0GHz', 'IsoFreq_Model_Oracle_4.0GHz',
    'IsoFreq_Oracle_1.0GHz', 'IsoFreq_Oracle_2.0GHz',
    'IsoFreq_Oracle_3.0GHz', 'IsoFreq_Oracle_4.0GHz',
    'Proactive_Hetero_Oracle',
]

HETERO_POLICIES = [
    'Static_P_3.0GHz', 'Static_E_3.0GHz',
    'EAS_Hetero', 'Micro_EAS', 'UCB1_Hetero',
    'EAS_With_DVFS', 'Threshold_Migration', 'Thread_Director',
    'Reactive_Combined_W1', 'Reactive_Combined_W5', 'Reactive_Combined_W10',
    'Model_Reactive_Hetero',
    'MPC_Oracle_Combined_W1', 'MPC_Oracle_Combined_W5', 'MPC_Oracle_Combined_W10',
    'EAS_Oracle_Hetero', 'Thread_Director_Oracle',
    'Model_Greedy_Oracle_Hetero',
    'Greedy_Oracle_Hetero',
    'IsoFreq_Oracle_3.0GHz', 'IsoFreq_Oracle_4.0GHz',
    'Proactive_Hetero_Oracle',
]

_CATEGORY_ORACLE = {
    'DVFS':    'Global_Oracle_P',
    'P_DVFS':  'Global_Oracle_P',
    'E_DVFS':  'Global_Oracle_E',
    'IsoFreq': 'Proactive_Hetero_Oracle',
    'Hetero':  'Proactive_Hetero_Oracle',
}

_HEURISTIC_CANDIDATES = {
    'DVFS':    ['Ondemand_P', 'Conservative_P', 'Schedutil_PELT_P',
                'Intel_HWP_P', 'EWMA_P', 'UCB1_P', 'Performance_Gov_P'],
    'P_DVFS':  ['Ondemand_P', 'Conservative_P', 'Schedutil_PELT_P',
                'Intel_HWP_P', 'EWMA_P', 'UCB1_P', 'Performance_Gov_P'],
    'E_DVFS':  ['Ondemand_E', 'Conservative_E', 'Schedutil_PELT_E', 'Intel_HWP_E', 'EWMA_E', 'UCB1_E', 'Performance_Gov_E'],
    'IsoFreq': ['EAS_Hetero', 'Micro_EAS', 'UCB1_Hetero'],
    'Hetero':  ['EAS_With_DVFS', 'Thread_Director', 'Threshold_Migration', 'EAS_Hetero'],
}

_HEURISTIC_ORACLE_CANDIDATES = {
    'DVFS':    ['Ondemand_Future_P', 'Conservative_Future_P', 'Schedutil_Future_P',
                'Intel_HWP_Future_P', 'EWMA_Future_P', 'UCB1_Future_P'],
    'P_DVFS':  ['Ondemand_Future_P', 'Conservative_Future_P', 'Schedutil_Future_P',
                'Intel_HWP_Future_P', 'EWMA_Future_P', 'UCB1_Future_P'],
    'E_DVFS':  ['Ondemand_Future_E', 'Conservative_Future_E', 'Schedutil_Future_E',
                'Intel_HWP_Future_E', 'EWMA_Future_E', 'UCB1_Future_E'],
    'IsoFreq': [f'IsoFreq_Oracle_Heuristic_{f}GHz' for f in ['1.0', '2.0', '3.0', '4.0']],
    'Hetero':  ['EAS_Oracle_Hetero', 'Thread_Director_Oracle'],
}

_CATEGORY_POLICIES = {
    'DVFS':    DVFS_POLICIES,
    'P_DVFS':  P_DVFS_POLICIES,
    'E_DVFS':  E_DVFS_POLICIES,
    'IsoFreq': ISOFREQ_POLICIES,
    'Hetero':  HETERO_POLICIES,
}

METRICS = ['EDP', 'ED2P']
CATEGORIES = ['DVFS', 'IsoFreq', 'Hetero']
BAR_CATEGORIES = ['P_DVFS', 'E_DVFS', 'IsoFreq', 'Hetero']  # used for bar plots


# ─────────────────────────────────────────────────────────────
# Suite classification
# ─────────────────────────────────────────────────────────────

def get_suite(wl_name):
    if wl_name.startswith('spec_5'):
        return 'SPEC2017'
    if wl_name.startswith('spec_6') or wl_name.startswith('spec_7') or wl_name.startswith('spec_8'):
        return 'SPEC2026'
    if wl_name.startswith('dacapo'):
        return 'DaCapo'
    return 'Other'


# ─────────────────────────────────────────────────────────────
# Color scheme
# ─────────────────────────────────────────────────────────────

def _policy_color(name):
    if 'Static' in name:             return '#bbbbbb'
    if 'Random' in name:             return '#999999'
    if 'Performance_Gov' in name:    return '#17becf'
    if 'Ondemand' in name:           return '#1f77b4'
    if 'Conservative' in name:       return '#4a9fd4'
    if 'Schedutil' in name:          return '#7ec6f0'
    if 'Intel_HWP' in name:         return '#aed6f0'
    if 'EWMA' in name:               return '#ff7f0e'
    if 'UCB1_Hetero' in name:        return '#d62728'
    if 'UCB1' in name:               return '#ffbb78'
    if 'Micro_EAS' in name:          return '#e377c2'
    if 'EAS_With_DVFS' in name:      return '#c5b0d5'
    if 'EAS_Hetero' in name:         return '#f7b6d2'
    if 'Threshold_Migration' in name: return '#9467bd'
    if 'Thread_Director' in name:    return '#7b4f9e'
    if 'Reactive_Combined' in name:  return '#bcbd22'
    if 'Reactive_Oracle' in name:           return '#2ca02c'
    if 'IsoFreq_Reactive_Oracle' in name:  return '#66bb6a'
    if 'IsoFreq_Oracle_Heuristic' in name: return '#a5d6a7'
    if 'IsoFreq_Model_Oracle' in name:     return '#80cbc4'
    if 'Model_IsoFreq' in name:            return '#4db6ac'
    if 'Greedy_Oracle_Hetero' in name:     return '#f48fb1'
    if 'EAS_Oracle' in name:              return '#f8bbd0'
    if 'Thread_Director_Oracle' in name:   return '#e91e63'
    if 'Ondemand_Oracle' in name:          return '#0d47a1'
    if 'Schedutil_Oracle' in name:         return '#1976d2'
    if 'EWMA_Oracle' in name:              return '#42a5f5'
    if 'Model_Global' in name:       return '#7b2d8b'
    if 'Model_MPC' in name:          return '#ab47bc'
    if 'Model_Greedy_Oracle' in name: return '#ba68c8'
    if 'Model_Greedy' in name:       return '#ce93d8'
    if 'IsoFreq_Oracle' in name:     return '#98df8a'
    if 'MPC_Oracle_Combined' in name: return '#8c564b'
    if 'Proactive_Hetero_Oracle' in name: return '#e41a1c'
    if 'Global_Oracle' in name:      return '#d62728'
    if 'MPC_Oracle' in name:         return '#e06c75'
    if 'Greedy_Oracle' in name:      return '#f7a19a'
    return '#7f7f7f'


def _short_name(name):
    """Multi-line display label for bar tick marks."""
    return (name
            .replace('Schedutil_PELT', 'Schedutil')
            .replace('Performance_Gov', 'Perf\nGov')
            .replace('Proactive_Hetero_Oracle', 'Hetero\nOracle')
            .replace('IsoFreq_Oracle_', 'IsoFreq\n')
            .replace('Reactive_Combined_', 'React\nComb\n')
            .replace('MPC_Oracle_Combined_', 'MPC\nComb\n')
            .replace('_P_', '_P\n')
            .replace('_E_', '_E\n')
            .replace('_', '\n'))


# ─────────────────────────────────────────────────────────────
# Core plotting helpers
# ─────────────────────────────────────────────────────────────

def _save_fig(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches='tight')
    plt.close(fig)


def _bar_plot(names, values, title, ylabel, outpath, oracle_line=None,
              figsize=None):
    """Generic normalized bar chart. NaN values shown as hatched placeholders."""
    n = len(names)
    if figsize is None:
        figsize = (max(14, n * 0.55 + 2), 6)

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(n)

    vals = np.array([v if v is not None and not (isinstance(v, float) and np.isnan(v))
                     else np.nan for v in values], dtype=float)
    valid = ~np.isnan(vals)

    colors = [_policy_color(nm) for nm in names]

    if valid.any():
        ax.bar(x[valid], vals[valid],
               color=[colors[i] for i in range(n) if valid[i]],
               edgecolor='black', linewidth=0.6)

    if (~valid).any():
        ax.bar(x[~valid], [0.05] * int((~valid).sum()),
               color='none', edgecolor='#aaaaaa', linewidth=1.2,
               linestyle='--', hatch='//')
        for i in np.where(~valid)[0]:
            ax.text(i, 0.07, '?', ha='center', va='bottom',
                    color='#aaaaaa', fontsize=8)

    if oracle_line is not None:
        ax.axhline(oracle_line, color='#e41a1c', linestyle='--',
                   linewidth=1.5, label=f'Oracle = {oracle_line:.2f}')
        ax.legend(fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([_short_name(nm) for nm in names],
                       fontsize=7, rotation=45, ha='right')
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=10, pad=8)
    ax.grid(axis='y', alpha=0.3)
    ax.set_xlim(-0.6, n - 0.4)

    _save_fig(fig, outpath)


# ─────────────────────────────────────────────────────────────
# Aggregation helpers
# ─────────────────────────────────────────────────────────────

def _phase_avg(df, policies):
    """Return per-workload per-metric per-policy mean (phases averaged)."""
    present = set(df['Policy'].unique())
    keep = [p for p in policies if p in present]
    sub = df[df['Policy'].isin(keep)]
    return (sub.groupby(['Workload', 'Metric', 'Policy'])['Final_Value']
               .mean()
               .reset_index()
               .rename(columns={'Final_Value': 'Mean_Value'}))


def _best_policy(candidates, ratio_lookup, prefer_low=True):
    """Return (name, value) of the best candidate from the lookup dict."""
    avail = {p: ratio_lookup[p] for p in candidates if p in ratio_lookup}
    if not avail:
        return None, np.nan
    key = min if prefer_low else max
    best = key(avail, key=avail.get)
    return best, avail[best]


# ─────────────────────────────────────────────────────────────
# Per-workload bar plots  (plots 1-6)
# ─────────────────────────────────────────────────────────────

def generate_workload_bars(df, output_dir):
    """Bar plots per workload (BAR_CATEGORIES × 2 metrics), phases averaged.

    P_DVFS and E_DVFS are plotted separately so P-core and E-core DVFS
    results are not visually merged. Normalizes to the category oracle.
    """
    out = Path(output_dir)
    bar_dir = out / 'bar_wl'
    csv_dir = out / 'csv'

    for cat in BAR_CATEGORIES:
        policies = _CATEGORY_POLICIES[cat]
        oracle = _CATEGORY_ORACLE[cat]

        avg_df = _phase_avg(df, policies)
        avg_df.to_csv(csv_dir / f'per_workload_{cat}.csv', index=False)

        for metric in METRICS:
            sub = avg_df[avg_df['Metric'] == metric]
            for wl in sorted(sub['Workload'].unique()):
                wl_sub = sub[sub['Workload'] == wl]
                lookup = dict(zip(wl_sub['Policy'], wl_sub['Mean_Value']))

                oracle_val = lookup.get(oracle, np.nan)
                if np.isnan(oracle_val) or oracle_val == 0:
                    continue

                present = [p for p in policies if p in lookup]
                norm_vals = [lookup[p] / oracle_val for p in present]

                _bar_plot(
                    present, norm_vals,
                    f'{wl} — {cat} {metric}  (norm. to {oracle})',
                    f'Norm. {metric} (oracle = 1.0)',
                    bar_dir / f'{cat}_{metric}_{wl}.png',
                    oracle_line=1.0,
                )


# ─────────────────────────────────────────────────────────────
# Suite-averaged bar plots  (plots 7-12)
# ─────────────────────────────────────────────────────────────

def generate_suite_bars(df, output_dir):
    """One bar plot per (suite, metric, category).

    Normalizes each workload to its own oracle, then averages ratios across
    workloads in the suite (equal workload weighting, avoids scale effects).
    """
    out = Path(output_dir)
    bar_dir = out / 'bar_suite'
    csv_dir = out / 'csv'

    for cat in BAR_CATEGORIES:
        policies = _CATEGORY_POLICIES[cat]
        oracle = _CATEGORY_ORACLE[cat]

        avg_df = _phase_avg(df, policies)
        avg_df['Suite'] = avg_df['Workload'].map(get_suite)
        csv_rows = []

        for metric in METRICS:
            sub = avg_df[avg_df['Metric'] == metric]
            for suite in sorted(sub['Suite'].unique()):
                suite_sub = sub[sub['Suite'] == suite]

                # Per-workload ratio to oracle → mean ratio across workloads
                policy_ratios = {}
                n_wl = 0
                for wl, wl_grp in suite_sub.groupby('Workload'):
                    lookup = dict(zip(wl_grp['Policy'], wl_grp['Mean_Value']))
                    ov = lookup.get(oracle, np.nan)
                    if np.isnan(ov) or ov == 0:
                        continue
                    n_wl += 1
                    for p, v in lookup.items():
                        policy_ratios.setdefault(p, []).append(v / ov)

                if n_wl == 0:
                    continue

                mean_ratios = {p: float(np.mean(vs)) for p, vs in policy_ratios.items()}

                present = [p for p in policies if p in mean_ratios]
                norm_vals = [mean_ratios[p] for p in present]

                csv_rows.extend([
                    {'Suite': suite, 'Metric': metric, 'Category': cat,
                     'Policy': p, 'Suite_Norm_Mean': mean_ratios[p]}
                    for p in present
                ])

                _bar_plot(
                    present, norm_vals,
                    (f'{suite} — {cat} {metric}  '
                     f'(norm. to {oracle}, n={n_wl} workloads)'),
                    f'Norm. {metric} (oracle = 1.0)',
                    bar_dir / f'{cat}_{metric}_{suite}.png',
                    oracle_line=1.0,
                )

        if csv_rows:
            pd.DataFrame(csv_rows).to_csv(
                csv_dir / f'suite_avg_{cat}.csv', index=False)


# ─────────────────────────────────────────────────────────────
# Taxonomy breakdown plots  (plots 13-18)
# ─────────────────────────────────────────────────────────────
#
# Taxonomy: 3 temporal modes × 3 decision axes = 9 bars per plot.
# Implemented bars have a real value; placeholders are hatched outlines.
#
# temporal:   Reactive  |  Workload Forecast  |  Perfect Future
# decision:   Heuristic | Model | Oracle
#
# Group colours (constant across categories):
#   Reactive      → blue  (#4e79a7)
#   Forecast      → orange (#f28e2b, dim for placeholders)
#   Perfect Future→ green  (#59a14f)

_GRP_COLOR = {
    0: '#4e79a7',  # Reactive Heuristic
    1: '#4e79a7',  # Reactive Model
    2: '#4e79a7',  # Reactive Oracle
    3: '#f28e2b',  # Forecast Heuristic  (placeholder)
    4: '#f28e2b',  # Forecast Model      (placeholder)
    5: '#f28e2b',  # Forecast Oracle     (placeholder)
    6: '#59a14f',  # Perfect Heuristic
    7: '#59a14f',  # Perfect Model
    8: '#59a14f',  # Perfect Oracle
    9: '#e41a1c',  # Global Oracle
}

# (label, policy_or_sentinel, is_placeholder)
# Sentinels starting with '__best_' are resolved at plot time to the best
# available policy from the corresponding candidate list.
_TAXONOMY_BARS = {
    'DVFS': [
        ('Reactive\nHeuristic\n(best)',      '__best_heuristic__',        False),
        ('Reactive\nModel\n(CatBoost)',      'Model_Greedy_P',            False),
        ('Reactive\nOracle\n(prior chunk)',  'Reactive_Oracle_P',         False),
        ('Forecast\nHeuristic',              None,                        True),
        ('Forecast\nModel',                  None,                        True),
        ('Forecast\nOracle',                 None,                        True),
        ('Perfect\nHeuristic\n(best)',       '__best_heuristic_oracle__', False),
        ('Perfect\nModel\n(CatBoost)',       'Model_Greedy_Oracle_P',     False),
        ('Perfect\nOracle\n(Greedy)',        'Greedy_Oracle_P',           False),
        ('Global\nOracle',                   'Global_Oracle_P',           False),
    ],
    'P_DVFS': [
        ('Reactive\nHeuristic\n(best)',      '__best_heuristic__',        False),
        ('Reactive\nModel\n(CatBoost)',      'Model_Greedy_P',            False),
        ('Reactive\nOracle\n(prior chunk)',  'Reactive_Oracle_P',         False),
        ('Forecast\nHeuristic',              None,                        True),
        ('Forecast\nModel',                  None,                        True),
        ('Forecast\nOracle',                 None,                        True),
        ('Perfect\nHeuristic\n(best)',       '__best_heuristic_oracle__', False),
        ('Perfect\nModel\n(CatBoost)',       'Model_Greedy_Oracle_P',     False),
        ('Perfect\nOracle\n(Greedy)',        'Greedy_Oracle_P',           False),
        ('Global\nOracle',                   'Global_Oracle_P',           False),
    ],
    'E_DVFS': [
        ('Reactive\nHeuristic\n(best)',      '__best_heuristic__',        False),
        ('Reactive\nModel\n(CatBoost)',      'Model_Greedy_E',            False),
        ('Reactive\nOracle\n(prior chunk)',  'Reactive_Oracle_E',         False),
        ('Forecast\nHeuristic',              None,                        True),
        ('Forecast\nModel',                  None,                        True),
        ('Forecast\nOracle',                 None,                        True),
        ('Perfect\nHeuristic\n(best)',       '__best_heuristic_oracle__', False),
        ('Perfect\nModel',                   None,                        True),
        ('Perfect\nOracle\n(Greedy)',        'Greedy_Oracle_E',           False),
        ('Global\nOracle',                   'Global_Oracle_E',           False),
    ],
    'IsoFreq': [
        # Reactive (bars 0-2)
        ('Reactive\nHeuristic\n(best)',      '__best_heuristic__',                False),
        ('Reactive\nModel\n(best freq)',     '__best_isofreq_model__',            False),
        ('Reactive\nOracle\n(best freq)',    '__best_isofreq_reactive_oracle__',  False),
        # Workload Forecast (bars 3-5): TBD
        ('Forecast\nHeuristic',              None,                                True),
        ('Forecast\nModel',                  None,                                True),
        ('Forecast\nOracle',                 None,                                True),
        # Perfect Future (bars 6-8)
        ('Perfect\nHeuristic\n(best freq)',  '__best_isofreq_oracle_heuristic__', False),
        ('Perfect\nModel\n(best freq)',      '__best_isofreq_model_oracle__',     False),
        ('Perfect\nOracle\n(best freq)',     '__best_isofreq_oracle__',           False),
        # Global (bar 9)
        ('Global\nOracle',                   'Proactive_Hetero_Oracle',           False),
    ],
    'Hetero': [
        # Reactive (bars 0-2)
        ('Reactive\nHeuristic\n(best)',      '__best_heuristic__',           False),
        ('Reactive\nModel\n(CatBoost)',      'Model_Reactive_Hetero',        False),
        ('Reactive\nOracle\n(prior chunk)',  'Reactive_Combined_W1',         False),
        # Workload Forecast (bars 3-5): TBD
        ('Forecast\nHeuristic',              None,                           True),
        ('Forecast\nModel',                  None,                           True),
        ('Forecast\nOracle',                 None,                           True),
        # Perfect Future (bars 6-8)
        ('Perfect\nHeuristic\n(best)',       '__best_hetero_oracle_heuristic__', False),
        ('Perfect\nModel\n(CatBoost)',       'Model_Greedy_Oracle_Hetero',   False),
        ('Perfect\nOracle\n(Greedy)',        'Greedy_Oracle_Hetero',         False),
        # Global (bar 9)
        ('Global\nOracle',                   'Proactive_Hetero_Oracle',      False),
    ],
}

_ISOFREQ_ORACLE_CANDIDATES = [f'IsoFreq_Oracle_{f}GHz'
                               for f in ['1.0', '2.0', '3.0', '4.0']]
_ISOFREQ_MODEL_CANDIDATES = [f'Model_IsoFreq_{f}GHz'
                               for f in ['1.0', '2.0', '3.0', '4.0']]
_ISOFREQ_REACTIVE_ORACLE_CANDIDATES = [f'IsoFreq_Reactive_Oracle_{f}GHz'
                                        for f in ['1.0', '2.0', '3.0', '4.0']]
_ISOFREQ_MODEL_ORACLE_CANDIDATES = [f'IsoFreq_Model_Oracle_{f}GHz'
                                     for f in ['1.0', '2.0', '3.0', '4.0']]
_ISOFREQ_ORACLE_HEURISTIC_CANDIDATES = [f'IsoFreq_Oracle_Heuristic_{f}GHz'
                                         for f in ['1.0', '2.0', '3.0', '4.0']]

_SUITE_DIR = {
    'SPEC2017': 'spec_2017',
    'SPEC2026': 'spec_2026',
    'DaCapo':   'dacapo',
    'Other':    'other',
}


def _suite_to_dir(suite):
    return _SUITE_DIR.get(suite, suite.lower().replace(' ', '_'))


def _build_sentinel_map(cat, ratios, heur_cands, heur_oracle_cands):
    """Resolve all sentinel policy names for a given ratio dict."""
    best_heur_name,        best_heur_val        = _best_policy(heur_cands,                        ratios)
    best_heur_ora_name,    best_heur_ora_val     = _best_policy(heur_oracle_cands,                 ratios)
    best_iso_name,         best_iso_val          = _best_policy(_ISOFREQ_ORACLE_CANDIDATES,         ratios)
    best_iso_model_name,   best_iso_model_val    = _best_policy(_ISOFREQ_MODEL_CANDIDATES,          ratios)
    best_iso_ro_name,      best_iso_ro_val       = _best_policy(_ISOFREQ_REACTIVE_ORACLE_CANDIDATES, ratios)
    best_iso_mo_name,      best_iso_mo_val       = _best_policy(_ISOFREQ_MODEL_ORACLE_CANDIDATES,   ratios)
    best_iso_oh_name,      best_iso_oh_val       = _best_policy(_ISOFREQ_ORACLE_HEURISTIC_CANDIDATES, ratios)
    return {
        '__best_heuristic__':               (best_heur_val,      best_heur_name),
        '__best_heuristic_oracle__':         (best_heur_ora_val,  best_heur_ora_name),
        '__best_isofreq_oracle__':           (best_iso_val,       best_iso_name),
        '__best_isofreq_model__':            (best_iso_model_val, best_iso_model_name),
        '__best_isofreq_reactive_oracle__':  (best_iso_ro_val,    best_iso_ro_name),
        '__best_isofreq_model_oracle__':     (best_iso_mo_val,    best_iso_mo_name),
        '__best_isofreq_oracle_heuristic__': (best_iso_oh_val,    best_iso_oh_name),
        '__best_hetero_oracle_heuristic__':  (best_heur_ora_val,  best_heur_ora_name),
    }


def _assemble_bars(bar_defs, ratios, sentinel_map):
    """Turn a list of (label, policy, is_placeholder) defs into bar arrays."""
    labels, vals, colors, placeholders, csv_rows = [], [], [], [], []
    for i, (lbl, policy, is_ph) in enumerate(bar_defs):
        color = _GRP_COLOR[i]
        if is_ph:
            labels.append(lbl)
            vals.append(np.nan)
            colors.append(color)
            placeholders.append(True)
            csv_rows.append({'Label': lbl, 'Policy': 'TBD', 'Norm_Value': np.nan, 'Placeholder': True})
            continue
        if policy in sentinel_map:
            val, pname = sentinel_map[policy]
            pname = pname or 'N/A'
            display = lbl + f'\n({pname})'
        else:
            val = ratios.get(policy, np.nan)
            pname = policy
            display = lbl
        labels.append(display)
        vals.append(val)
        colors.append(color)
        placeholders.append(False)
        csv_rows.append({'Label': lbl, 'Policy': pname, 'Norm_Value': val, 'Placeholder': False})
    return labels, vals, colors, placeholders, csv_rows


def generate_taxonomy(df, output_dir):
    """Suite-level taxonomy bar charts, saved under taxonomy/suite_level/.

    Normalizes per workload to the category oracle, averages across workloads.
    Unimplemented bars shown as hatched placeholders.
    """
    out = Path(output_dir)
    tax_dir = out / 'taxonomy' / 'suite_level'
    csv_dir = out / 'csv'
    tax_dir.mkdir(parents=True, exist_ok=True)

    for cat in BAR_CATEGORIES:
        if cat not in _TAXONOMY_BARS:
            continue
        oracle = _CATEGORY_ORACLE[cat]
        heur_cands = _HEURISTIC_CANDIDATES[cat]
        heur_oracle_cands = _HEURISTIC_ORACLE_CANDIDATES[cat]

        all_needed = (set(_CATEGORY_POLICIES[cat]) | set(heur_cands)
                      | set(heur_oracle_cands)
                      | set(_ISOFREQ_ORACLE_CANDIDATES) | set(_ISOFREQ_MODEL_CANDIDATES)
                      | set(_ISOFREQ_REACTIVE_ORACLE_CANDIDATES)
                      | set(_ISOFREQ_MODEL_ORACLE_CANDIDATES)
                      | set(_ISOFREQ_ORACLE_HEURISTIC_CANDIDATES))
        avg_df = _phase_avg(df, list(all_needed))
        avg_df['Suite'] = avg_df['Workload'].map(get_suite)

        for metric in METRICS:
            sub = avg_df[avg_df['Metric'] == metric]
            for suite in sorted(sub['Suite'].unique()):
                suite_sub = sub[sub['Suite'] == suite]

                policy_ratios = {}
                n_wl = 0
                for wl, wl_grp in suite_sub.groupby('Workload'):
                    lookup = dict(zip(wl_grp['Policy'], wl_grp['Mean_Value']))
                    ov = lookup.get(oracle, np.nan)
                    if np.isnan(ov) or ov == 0:
                        continue
                    n_wl += 1
                    for p, v in lookup.items():
                        policy_ratios.setdefault(p, []).append(v / ov)

                if n_wl == 0:
                    continue

                mean_ratios = {p: float(np.mean(vs)) for p, vs in policy_ratios.items()}
                sentinel_map = _build_sentinel_map(cat, mean_ratios, heur_cands, heur_oracle_cands)
                labels, vals, colors, phs, csv_rows = _assemble_bars(
                    _TAXONOMY_BARS[cat], mean_ratios, sentinel_map)

                _taxonomy_bar_plot(
                    labels, vals, colors, phs,
                    f'{suite} — {cat} {metric} Taxonomy  (norm. to {oracle}, n={n_wl} workloads)',
                    tax_dir / f'{cat}_{metric}_{suite}.png',
                )
                pd.DataFrame(csv_rows).to_csv(
                    csv_dir / f'taxonomy_{metric}_{cat}_{suite}.csv', index=False)


def generate_per_workload_taxonomy(df, output_dir):
    """Per-workload taxonomy bar charts, saved under taxonomy/{suite}/per_workload/.

    Normalizes each policy to the category oracle for that single workload
    (phases averaged), so inter-workload differences are fully visible.
    """
    out = Path(output_dir)
    csv_dir = out / 'csv'

    for cat in BAR_CATEGORIES:
        if cat not in _TAXONOMY_BARS:
            continue
        oracle = _CATEGORY_ORACLE[cat]
        heur_cands = _HEURISTIC_CANDIDATES[cat]
        heur_oracle_cands = _HEURISTIC_ORACLE_CANDIDATES[cat]

        all_needed = (set(_CATEGORY_POLICIES[cat]) | set(heur_cands)
                      | set(heur_oracle_cands)
                      | set(_ISOFREQ_ORACLE_CANDIDATES) | set(_ISOFREQ_MODEL_CANDIDATES)
                      | set(_ISOFREQ_REACTIVE_ORACLE_CANDIDATES)
                      | set(_ISOFREQ_MODEL_ORACLE_CANDIDATES)
                      | set(_ISOFREQ_ORACLE_HEURISTIC_CANDIDATES))
        avg_df = _phase_avg(df, list(all_needed))

        for metric in METRICS:
            sub = avg_df[avg_df['Metric'] == metric]
            for wl, wl_grp in sub.groupby('Workload'):
                lookup = dict(zip(wl_grp['Policy'], wl_grp['Mean_Value']))
                ov = lookup.get(oracle, np.nan)
                if np.isnan(ov) or ov == 0:
                    continue

                ratios = {p: v / ov for p, v in lookup.items()}
                sentinel_map = _build_sentinel_map(cat, ratios, heur_cands, heur_oracle_cands)
                labels, vals, colors, phs, csv_rows = _assemble_bars(
                    _TAXONOMY_BARS[cat], ratios, sentinel_map)

                suite_dir = _suite_to_dir(get_suite(wl))
                wl_tax_dir = out / 'taxonomy' / suite_dir / 'per_workload' / cat
                wl_tax_dir.mkdir(parents=True, exist_ok=True)

                _taxonomy_bar_plot(
                    labels, vals, colors, phs,
                    f'{wl} — {cat} {metric}  (norm. to {oracle})',
                    wl_tax_dir / f'{metric}_{wl}.png',
                )
                pd.DataFrame(csv_rows).to_csv(
                    csv_dir / f'taxonomy_{metric}_{cat}_{wl}.csv', index=False)


def _taxonomy_bar_plot(labels, heights, colors, is_placeholder, title, outpath):
    """Taxonomy bar chart with 3 temporal group dividers."""
    n = len(labels)
    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(n)

    # Draw bars
    for i in range(n):
        h = heights[i]
        have_val = h is not None and not (isinstance(h, float) and np.isnan(h))

        if is_placeholder[i] or not have_val:
            ax.bar(x[i], 0.06, color='none', edgecolor='#aaaaaa',
                   linewidth=1.4, linestyle='--', hatch='//')
            ax.text(x[i], 0.09, 'TBD', ha='center', va='bottom',
                    color='#aaaaaa', fontsize=8)
        else:
            alpha = 0.35 if is_placeholder[i] else 1.0
            ax.bar(x[i], h, color=colors[i], alpha=alpha,
                   edgecolor='black', linewidth=0.8)
            ax.text(x[i], h + 0.008, f'{h:.3f}',
                    ha='center', va='bottom', fontsize=8, fontweight='bold')

    # Oracle reference line at 1.0
    ax.axhline(1.0, color='#e41a1c', linestyle='--',
               linewidth=2, label='Oracle = 1.0')

    # Temporal group separators and headers
    valid_heights = [h for h in heights
                     if h is not None and not (isinstance(h, float) and np.isnan(h))]
    y_top = max(max(valid_heights) * 1.2, 1.3) if valid_heights else 1.5
    ax.set_ylim(0, y_top)

    for sep_x in [2.5, 5.5, 8.5]:
        ax.axvline(sep_x, color='#cccccc', linewidth=1.5, linestyle=':')

    group_midpoints = [
        (0, 2, 'Reactive',        '#444444'),
        (3, 5, 'Workload\nForecast', '#444444'),
        (6, 8, 'Perfect\nFuture', '#444444'),
        (9, 9, 'Global\nBound',   '#e41a1c'),
    ]
    for start, end, grp_lbl, grp_color in group_midpoints:
        ax.text((start + end) / 2, y_top * 0.97, grp_lbl,
                ha='center', va='top', fontsize=9,
                color=grp_color, fontstyle='italic')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(f'Norm. EDP/ED²P  (lower = better)', fontsize=9)
    ax.set_title(title, fontsize=10, pad=10)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.set_xlim(-0.6, n - 0.4)

    _save_fig(fig, outpath)


ISOFREQ_FREQ_LABELS = ['1.0GHz', '2.0GHz', '3.0GHz', '4.0GHz']

_FREQ_DISPLAY = {'1.0GHz': '1 GHz', '2.0GHz': '2 GHz', '3.0GHz': '3 GHz', '4.0GHz': '4 GHz'}


def _isofreq_per_freq_static_candidates(freq):
    """Best static core-type selection at a fixed frequency.

    These are the heuristic candidates for per-frequency taxonomy plots.
    Using frequency-adaptive policies (EAS_Hetero etc.) as the heuristic
    baseline is apples-to-oranges: they can choose any frequency and would
    trivially beat a fixed-frequency oracle. Static P vs E at the same
    frequency is the correct within-constraint comparison.
    """
    return [f'Static_P_{freq}', f'Static_E_{freq}']


def _isofreq_per_freq_bars(freq):
    """10-bar taxonomy for a single fixed frequency.

    Bar 0 uses '__best_static_{freq}__' sentinel — resolved to best of
    Static_P_{freq} / Static_E_{freq}. This avoids the apples-to-oranges
    issue where EAS_Hetero (frequency-adaptive) beats a fixed-freq oracle.
    """
    return [
        (f'Static\nBaseline\n(best core)',      f'__best_static_{freq}__',            False),
        (f'Reactive\nModel\n({freq})',           f'Model_IsoFreq_{freq}',              False),
        (f'Reactive\nOracle\n({freq})',          f'IsoFreq_Reactive_Oracle_{freq}',    False),
        ('Forecast\nHeuristic',                  None,                                 True),
        ('Forecast\nModel',                      None,                                 True),
        ('Forecast\nOracle',                     None,                                 True),
        (f'Perfect\nHeuristic\n({freq})',        f'IsoFreq_Oracle_Heuristic_{freq}',   False),
        (f'Perfect\nModel\n({freq})',            f'IsoFreq_Model_Oracle_{freq}',       False),
        (f'Perfect\nOracle\n({freq})',           f'IsoFreq_Oracle_{freq}',             False),
        ('Global\nOracle',                       'Proactive_Hetero_Oracle',            False),
    ]


def _isofreq_freq_sentinel_map(freq, ratios):
    """Sentinel map for a per-frequency IsoFreq taxonomy.

    Resolves __best_static_{freq}__ to the better of Static_P_{freq} /
    Static_E_{freq} — the correct within-frequency-constraint baseline.
    """
    static_cands = _isofreq_per_freq_static_candidates(freq)
    best_name, best_val = _best_policy(static_cands, ratios)
    return {f'__best_static_{freq}__': (best_val, best_name)}


def _isofreq_freq_all_needed(freq):
    """All policy names needed for a per-frequency IsoFreq taxonomy plot."""
    return (set(_isofreq_per_freq_static_candidates(freq))
            | {f'IsoFreq_Oracle_{freq}', 'Proactive_Hetero_Oracle',
               f'Model_IsoFreq_{freq}', f'IsoFreq_Reactive_Oracle_{freq}',
               f'IsoFreq_Oracle_Heuristic_{freq}', f'IsoFreq_Model_Oracle_{freq}'})


def generate_isofreq_per_freq_taxonomy(df, output_dir):
    """Suite-level per-frequency IsoFreq taxonomy, under taxonomy/suite_level/.

    Bar 0 is the better static core type at that frequency (not EAS_Hetero,
    which is frequency-adaptive and would trivially beat a fixed-freq oracle).
    Global Oracle bar may fall below 1.0 since Proactive_Hetero_Oracle can
    select any frequency.
    """
    out = Path(output_dir)
    tax_dir = out / 'taxonomy' / 'suite_level'
    csv_dir = out / 'csv'
    tax_dir.mkdir(parents=True, exist_ok=True)

    for freq in ISOFREQ_FREQ_LABELS:
        oracle = f'IsoFreq_Oracle_{freq}'
        freq_bar_defs = _isofreq_per_freq_bars(freq)
        avg_df = _phase_avg(df, list(_isofreq_freq_all_needed(freq)))
        avg_df['Suite'] = avg_df['Workload'].map(get_suite)

        for metric in METRICS:
            sub = avg_df[avg_df['Metric'] == metric]
            for suite in sorted(sub['Suite'].unique()):
                policy_ratios = {}
                n_wl = 0
                for wl, wl_grp in sub[sub['Suite'] == suite].groupby('Workload'):
                    lookup = dict(zip(wl_grp['Policy'], wl_grp['Mean_Value']))
                    ov = lookup.get(oracle, np.nan)
                    if np.isnan(ov) or ov == 0:
                        continue
                    n_wl += 1
                    for p, v in lookup.items():
                        policy_ratios.setdefault(p, []).append(v / ov)
                if n_wl == 0:
                    continue

                mean_ratios = {p: float(np.mean(vs)) for p, vs in policy_ratios.items()}
                sentinel_map = _isofreq_freq_sentinel_map(freq, mean_ratios)
                labels, vals, colors, phs, csv_rows = _assemble_bars(
                    freq_bar_defs, mean_ratios, sentinel_map)

                _taxonomy_bar_plot(
                    labels, vals, colors, phs,
                    f'{suite} — IsoFreq {_FREQ_DISPLAY[freq]} {metric} Taxonomy'
                    f'  (norm. to {oracle}, n={n_wl} workloads)',
                    tax_dir / f'IsoFreq_{freq}_{metric}_{suite}.png',
                )
                pd.DataFrame(csv_rows).to_csv(
                    csv_dir / f'taxonomy_{metric}_IsoFreq_{freq}_{suite}.csv', index=False)


def generate_per_workload_isofreq_taxonomy(df, output_dir):
    """Per-workload per-frequency IsoFreq taxonomy.

    Output: taxonomy/{suite_dir}/per_workload/IsoFreq/{freq}_{metric}_{wl}.png
    """
    out = Path(output_dir)
    csv_dir = out / 'csv'

    for freq in ISOFREQ_FREQ_LABELS:
        oracle = f'IsoFreq_Oracle_{freq}'
        freq_bar_defs = _isofreq_per_freq_bars(freq)
        avg_df = _phase_avg(df, list(_isofreq_freq_all_needed(freq)))

        for metric in METRICS:
            sub = avg_df[avg_df['Metric'] == metric]
            for wl, wl_grp in sub.groupby('Workload'):
                lookup = dict(zip(wl_grp['Policy'], wl_grp['Mean_Value']))
                ov = lookup.get(oracle, np.nan)
                if np.isnan(ov) or ov == 0:
                    continue

                ratios = {p: v / ov for p, v in lookup.items()}
                sentinel_map = _isofreq_freq_sentinel_map(freq, ratios)
                labels, vals, colors, phs, _ = _assemble_bars(
                    freq_bar_defs, ratios, sentinel_map)

                suite_dir = _suite_to_dir(get_suite(wl))
                wl_tax_dir = out / 'taxonomy' / suite_dir / 'per_workload' / 'IsoFreq'
                wl_tax_dir.mkdir(parents=True, exist_ok=True)

                _taxonomy_bar_plot(
                    labels, vals, colors, phs,
                    f'{wl} — IsoFreq {_FREQ_DISPLAY[freq]} {metric}  (norm. to {oracle})',
                    wl_tax_dir / f'{freq}_{metric}_{wl}.png',
                )


# ─────────────────────────────────────────────────────────────
# CSV dumps
# ─────────────────────────────────────────────────────────────

def dump_summary_csv(df, output_dir):
    """Write all_phases_summary.csv and derived per-workload / suite CSVs."""
    out = Path(output_dir)
    csv_dir = out / 'csv'
    csv_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(out / 'all_phases_summary.csv', index=False)

    wl_avg = (df.groupby(['Workload', 'Metric', 'Policy'])['Final_Value']
                .mean()
                .reset_index()
                .rename(columns={'Final_Value': 'Workload_Mean'}))
    wl_avg.to_csv(csv_dir / 'per_workload_avg.csv', index=False)

    wl_avg['Suite'] = wl_avg['Workload'].map(get_suite)
    suite_avg = (wl_avg.groupby(['Suite', 'Metric', 'Policy'])['Workload_Mean']
                       .mean()
                       .reset_index()
                       .rename(columns={'Workload_Mean': 'Suite_Mean'}))
    suite_avg.to_csv(csv_dir / 'suite_avg_all_policies.csv', index=False)


# ─────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────

def generate_all_plots(df, output_dir):
    """Generate all 18 plot types and CSVs from the summary DataFrame.

    df: pandas DataFrame with columns [Workload, Phase, Metric, Policy, Final_Value]
    output_dir: root output directory (string or Path)
    """
    out = Path(output_dir)
    (out / 'csv').mkdir(parents=True, exist_ok=True)

    dump_summary_csv(df, out)
    generate_workload_bars(df, out)
    generate_suite_bars(df, out)
    generate_taxonomy(df, out)
    generate_per_workload_taxonomy(df, out)
    generate_isofreq_per_freq_taxonomy(df, out)
    generate_per_workload_isofreq_taxonomy(df, out)
    print(f"Plots and CSVs saved to {out}")

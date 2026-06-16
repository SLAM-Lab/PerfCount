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
    taxonomy/DVFS_{metric}_{suite}.png
    taxonomy/IsoFreq_{metric}_{suite}.png
    taxonomy/Hetero_{metric}_{suite}.png

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

DVFS_POLICIES = [
    # P-core statics
    'Static_P_1.0GHz', 'Static_P_2.0GHz', 'Static_P_3.0GHz', 'Static_P_4.0GHz',
    # P-core heuristic governors
    'Random_P', 'Performance_Gov_P', 'Ondemand_P', 'Conservative_P',
    'Schedutil_PELT_P', 'Intel_HWP_P', 'EWMA_P', 'UCB1_P',
    # P-core reactive oracle + model + perfect-future oracle
    'Reactive_Oracle_P',
    'Model_Greedy_P', 'Model_Global_P',
    'Greedy_Oracle_P', 'MPC_Oracle_P_W5', 'MPC_Oracle_P_W10', 'Global_Oracle_P',
    # E-core statics
    'Static_E_1.0GHz', 'Static_E_2.0GHz', 'Static_E_3.0GHz', 'Static_E_4.0GHz',
    # E-core heuristic governors
    'Random_E', 'Ondemand_E', 'Conservative_E', 'Schedutil_PELT_E', 'EWMA_E', 'UCB1_E',
    # E-core reactive oracle + perfect-future oracle
    'Reactive_Oracle_E',
    'Greedy_Oracle_E', 'MPC_Oracle_E_W5', 'MPC_Oracle_E_W10', 'Global_Oracle_E',
]

ISOFREQ_POLICIES = [
    'Static_P_1.0GHz', 'Static_E_1.0GHz',
    'Static_P_2.0GHz', 'Static_E_2.0GHz',
    'Static_P_3.0GHz', 'Static_E_3.0GHz',
    'Static_P_4.0GHz', 'Static_E_4.0GHz',
    'Micro_EAS', 'EAS_Hetero', 'UCB1_Hetero',
    'IsoFreq_Oracle_1.0GHz', 'IsoFreq_Oracle_2.0GHz',
    'IsoFreq_Oracle_3.0GHz', 'IsoFreq_Oracle_4.0GHz',
    'Proactive_Hetero_Oracle',
]

HETERO_POLICIES = [
    'Static_P_3.0GHz', 'Static_E_3.0GHz',
    'EAS_Hetero', 'Micro_EAS', 'UCB1_Hetero',
    'EAS_With_DVFS', 'Threshold_Migration', 'Thread_Director',
    'Reactive_Combined_W1', 'Reactive_Combined_W5', 'Reactive_Combined_W10',
    'MPC_Oracle_Combined_W1', 'MPC_Oracle_Combined_W5', 'MPC_Oracle_Combined_W10',
    'IsoFreq_Oracle_3.0GHz', 'IsoFreq_Oracle_4.0GHz',
    'Proactive_Hetero_Oracle',
]

_CATEGORY_ORACLE = {
    'DVFS':    'Global_Oracle_P',
    'IsoFreq': 'Proactive_Hetero_Oracle',
    'Hetero':  'Proactive_Hetero_Oracle',
}

_HEURISTIC_CANDIDATES = {
    'DVFS':    ['Ondemand_P', 'Conservative_P', 'Schedutil_PELT_P',
                'Intel_HWP_P', 'EWMA_P', 'UCB1_P', 'Performance_Gov_P'],
    'IsoFreq': ['EAS_Hetero', 'Micro_EAS', 'UCB1_Hetero'],
    'Hetero':  ['EAS_With_DVFS', 'Thread_Director', 'Threshold_Migration', 'EAS_Hetero'],
}

_CATEGORY_POLICIES = {
    'DVFS':    DVFS_POLICIES,
    'IsoFreq': ISOFREQ_POLICIES,
    'Hetero':  HETERO_POLICIES,
}

METRICS = ['EDP', 'ED2P']
CATEGORIES = ['DVFS', 'IsoFreq', 'Hetero']


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
    if 'Reactive_1_Step' in name:    return '#2ca02c'
    if 'Model_Global' in name:       return '#7b2d8b'
    if 'Model_MPC' in name:          return '#ab47bc'
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
    """Six bar plots per workload (3 categories × 2 metrics), phases averaged.

    Normalizes each policy to the category oracle before plotting so bars
    are dimensionless ratios (oracle = 1.0).
    """
    out = Path(output_dir)
    bar_dir = out / 'bar_wl'
    csv_dir = out / 'csv'

    for cat in CATEGORIES:
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

    for cat in CATEGORIES:
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
    6: '#59a14f',  # Perfect Heuristic   (placeholder)
    7: '#59a14f',  # Perfect Model       (placeholder)
    8: '#59a14f',  # Perfect Oracle
}

# (label, policy_or_sentinel, is_placeholder)
# Sentinels: '__best_heuristic__', '__best_isofreq__'
_TAXONOMY_BARS = {
    'DVFS': [
        # Reactive: only data from prior chunks (causal, deployable)
        ('Reactive\nHeuristic\n(best)',     '__best_heuristic__', False),
        ('Reactive\nModel\n(CatBoost)',     'Model_Greedy_P',     False),
        ('Reactive\nOracle\n(prior chunk)', 'Reactive_Oracle_P',  False),
        # Workload Forecast: model predicts future chunk behavior (not yet implemented)
        ('Forecast\nHeuristic',             None,                 True),
        ('Forecast\nModel',                 None,                 True),
        ('Forecast\nOracle',                None,                 True),
        # Perfect Future: sees current/all future chunk data (not deployable)
        ('Perfect\nHeuristic',              None,                 True),
        ('Perfect\nModel\n(Global)',        'Model_Global_P',     False),
        ('Perfect\nOracle\n(Global)',       'Global_Oracle_P',    False),
    ],
    'IsoFreq': [
        # Reactive: only prior-chunk data
        ('Reactive\nHeuristic\n(best)',   '__best_heuristic__',  False),
        ('Reactive\nModel\n(cross-proc)', None,                  True),
        ('Reactive\nOracle',              None,                  True),
        # Workload Forecast
        ('Forecast\nHeuristic',           None,                  True),
        ('Forecast\nModel',               None,                  True),
        ('Forecast\nOracle',              None,                  True),
        # Perfect Future: global Viterbi over all chunks (not deployable)
        ('Perfect\nHeuristic',            None,                  True),
        ('Perfect\nOracle\n(IsoFreq)',    '__best_isofreq__',    False),
        ('Perfect\nOracle\n(Full)',       'Proactive_Hetero_Oracle', False),
    ],
    'Hetero': [
        # Reactive: only prior-chunk data
        ('Reactive\nHeuristic\n(best)',  '__best_heuristic__',      False),
        ('Reactive\nModel',              None,                       True),
        ('Reactive\nOracle',             'Reactive_Combined_W1',     False),
        # Workload Forecast
        ('Forecast\nHeuristic',          None,                       True),
        ('Forecast\nModel',              None,                       True),
        ('Forecast\nOracle',             None,                       True),
        # Perfect Future: sees current/future chunk data (not deployable)
        ('Perfect\nHeuristic',           None,                       True),
        ('Perfect\nOracle\n(MPC-W1)',    'MPC_Oracle_Combined_W1',  False),
        ('Perfect\nOracle\n(Full)',      'Proactive_Hetero_Oracle',  False),
    ],
}

_ISOFREQ_ORACLE_CANDIDATES = [f'IsoFreq_Oracle_{f}GHz'
                               for f in ['1.0', '2.0', '3.0', '4.0']]


def generate_taxonomy(df, output_dir):
    """Per-suite taxonomy bar charts (plots 13-18).

    Normalizes per workload to the category oracle, averages across workloads.
    Unimplemented bars shown as hatched placeholders.
    """
    out = Path(output_dir)
    tax_dir = out / 'taxonomy'
    csv_dir = out / 'csv'

    for cat in CATEGORIES:
        oracle = _CATEGORY_ORACLE[cat]
        heur_cands = _HEURISTIC_CANDIDATES[cat]

        # Gather all policies needed (implemented bars + heuristic candidates)
        all_needed = set(_CATEGORY_POLICIES[cat]) | set(heur_cands) | set(_ISOFREQ_ORACLE_CANDIDATES)
        avg_df = _phase_avg(df, list(all_needed))
        avg_df['Suite'] = avg_df['Workload'].map(get_suite)

        for metric in METRICS:
            sub = avg_df[avg_df['Metric'] == metric]
            for suite in sorted(sub['Suite'].unique()):
                suite_sub = sub[sub['Suite'] == suite]

                # Per-workload normalization → mean across workloads
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

                best_heur_name, best_heur_val = _best_policy(heur_cands, mean_ratios)
                best_iso_name, best_iso_val = _best_policy(_ISOFREQ_ORACLE_CANDIDATES, mean_ratios)

                bar_labels, bar_vals, bar_colors, bar_placeholder = [], [], [], []
                csv_rows = []

                for i, (lbl, policy, is_ph) in enumerate(_TAXONOMY_BARS[cat]):
                    color = _GRP_COLOR[i]

                    if is_ph:
                        bar_labels.append(lbl)
                        bar_vals.append(np.nan)
                        bar_colors.append(color)
                        bar_placeholder.append(True)
                        csv_rows.append({'Label': lbl, 'Policy': 'TBD',
                                         'Norm_Mean': np.nan, 'Placeholder': True})
                        continue

                    if policy == '__best_heuristic__':
                        val = best_heur_val
                        pname = best_heur_name or 'N/A'
                        display = lbl + f'\n({pname})'
                    elif policy == '__best_isofreq__':
                        val = best_iso_val
                        pname = best_iso_name or 'N/A'
                        display = lbl + f'\n({pname})'
                    else:
                        val = mean_ratios.get(policy, np.nan)
                        pname = policy
                        display = lbl

                    bar_labels.append(display)
                    bar_vals.append(val)
                    bar_colors.append(color)
                    bar_placeholder.append(False)
                    csv_rows.append({'Label': lbl, 'Policy': pname,
                                     'Norm_Mean': val, 'Placeholder': False})

                _taxonomy_bar_plot(
                    bar_labels, bar_vals, bar_colors, bar_placeholder,
                    (f'{suite} — {cat} {metric} Taxonomy'
                     f'  (norm. to {oracle}, n={n_wl} workloads)'),
                    tax_dir / f'{cat}_{metric}_{suite}.png',
                )
                pd.DataFrame(csv_rows).to_csv(
                    csv_dir / f'taxonomy_{metric}_{cat}_{suite}.csv', index=False)


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

    for sep_x in [2.5, 5.5]:
        ax.axvline(sep_x, color='#cccccc', linewidth=1.5, linestyle=':')

    group_midpoints = [(0, 2, 'Reactive'), (3, 5, 'Workload\nForecast'), (6, 8, 'Perfect\nFuture')]
    for start, end, grp_lbl in group_midpoints:
        ax.text((start + end) / 2, y_top * 0.97, grp_lbl,
                ha='center', va='top', fontsize=9,
                color='#444444', fontstyle='italic')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(f'Norm. EDP/ED²P  (lower = better)', fontsize=9)
    ax.set_title(title, fontsize=10, pad=10)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.set_xlim(-0.6, n - 0.4)

    _save_fig(fig, outpath)


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
    print(f"Plots and CSVs saved to {out}")

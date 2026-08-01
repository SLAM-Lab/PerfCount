"""Evaluating a policy: the dispatch that decides how each one is scored.

`process_workload` in main.py builds a large declarative registry of policies (roughly 150 of
them, as {name: (callable, args)}). This module owns the other half: taking that registry and
producing a cost trace per policy per metric.

That sounds mechanical, and it is not. A policy is scored by one of four paths, chosen from
attributes on the callable, and **the paths do not charge the same costs**:

  metric_independent   one action sequence scored under both metrics (heuristics whose choice
                       does not depend on the objective). Warmup IS applied.
  is_viterbi_oracle    full-trace DP. Expensive, so the ACTION PATH is cached and re-scored;
                       the DP is warmup-unaware by design, so the path is metric- and
                       warmup-stable and only the re-scoring differs.
  returns_actions      the policy can report its chosen configurations, so the trace is
                       recomputed with the cross-cluster cache-warmup penalty applied.
  (default)            the policy's own returned trace is used VERBATIM -- no warmup penalty,
                       no diagnostics row, no action dump.

A migrating policy that lands in the default path is therefore not charged for its migrations
while its competitors are. This is not a hypothetical failure: the deployable heterogeneous gate
sat in that path for the whole study, went 55,048 P<->E migrations unpaid, and scored below the
global oracle under ED2P -- structurally impossible, and misread for a long time as a defect in
the oracle. See POLICIES.md and utils/test_policy_contract.py, which fails the build if any
action-capable policy omits the declaration.

The evaluator is constructed per workload-phase because nearly all of its state is
phase-specific (the traces, the transition matrices, the identity used in dump filenames).
"""
import csv
import json
import os
import time
from pathlib import Path

import numpy as np

import warmup_model
from decision_policies import compute_trace_stats, accumulate_trace


def call_with_stats(policy_fn, *args, _acts_out=None, **kwargs):
    """Call a policy, returning (trace, stats | None).

    `_acts_out`, when a list is given, receives (actions, local_names) so the caller can dump
    the per-chunk decision sequence. Policies that cannot report actions return stats of None,
    which is why they are also absent from diagnostics.csv -- a useful symptom when diagnosing
    a policy that has silently taken the verbatim path.
    """
    if getattr(policy_fn, 'returns_actions', False):
        trace, actions, local_names = policy_fn(*args, _return_actions=True, **kwargs)
        if _acts_out is not None:
            _acts_out.append((actions, local_names))
        return trace, compute_trace_stats(list(actions), local_names)
    return policy_fn(*args, **kwargs), None


def record_diag(diag_results, wl, ph, metric, name, stats):
    """Append one diagnostics row. Policies with no stats (see above) are skipped."""
    if stats is None:
        return
    diag_results.append({
        'Workload': wl, 'Phase': ph, 'Metric': metric, 'Policy': name,
        **{k: v for k, v in stats.items() if not k.startswith('frac_')},
        'config_fracs': json.dumps({k[5:]: round(v, 4)
                                    for k, v in stats.items() if k.startswith('frac_')}),
    })


class PolicyEvaluator:
    """Scores a registry of policies for one workload-phase.

    Parameters mirror what a policy is charged against: the per-chunk traces, the transition
    cost matrices, and the warmup model. `warmup` is the (A_PtoE, tau_PtoE, A_EtoP, tau_EtoP, K)
    tuple; `apply_warmup` gates whether it is charged at all.
    """

    def __init__(self, wl, ph, configs, time_mat, energy_mat, trans_lat, trans_nrg,
                 metrics, diag_results, apply_warmup, warmup,
                 viterbi_cache_dir=None, env_tag='', pred_tag=''):
        self.wl, self.ph = wl, ph
        self.configs = configs
        self.time_mat, self.energy_mat = time_mat, energy_mat
        self.trans_lat, self.trans_nrg = trans_lat, trans_nrg
        self.metrics = metrics
        self.diag_results = diag_results
        self.apply_warmup = apply_warmup
        self.warmup = warmup
        self.viterbi_cache_dir = viterbi_cache_dir
        self.env_tag, self.pred_tag = env_tag, pred_tag
        # Per-chunk decision dump, opt-in via the environment so a normal run pays nothing.
        self._dump_dir = os.environ.get('DUMP_ACTIONS_DIR')
        self._dump_pols = {p for p in os.environ.get('DUMP_ACTIONS_POLICIES', '').split(',') if p}
        self._time_pol = os.environ.get('SIM_TIME_POLICIES')

    # -- cost of a known action path -------------------------------------------------
    def _warmup_active(self):
        a_pte, _, a_etp, _, _ = self.warmup
        return self.apply_warmup and not (a_pte == 0.0 and a_etp == 0.0)

    def trace_from_path(self, actions, local_names, metric):
        """Cost trace for an action path, charging warmup if enabled.

        Separate from `apply_warmup_trace` so a cached Viterbi path can be re-scored without
        re-running the full-trace DP.
        """
        idx = [self.configs.index(c) for c in local_names]
        t_sub = self.time_mat[:, idx]
        e_sub = self.energy_mat[:, idx]
        lat_sub = self.trans_lat[np.ix_(idx, idx)]
        nrg_sub = self.trans_nrg[np.ix_(idx, idx)]
        if self._warmup_active():
            a_pte, tau_pte, a_etp, tau_etp, k = self.warmup
            t_sub = warmup_model.apply_warmup_penalty(
                t_sub, actions, local_names, a_pte, tau_pte, a_etp, tau_etp, k)
        return accumulate_trace(t_sub, e_sub, lat_sub, nrg_sub, actions, metric=metric)

    def apply_warmup_trace(self, fn, args, metric):
        """Run a policy, then recompute its trace with the warmup penalty charged."""
        trace, actions, local_names = fn(*args, _return_actions=True, metric=metric)
        stats = compute_trace_stats(list(actions), local_names)
        if not self._warmup_active():
            return trace, stats, actions, local_names
        return self.trace_from_path(actions, local_names, metric), stats, actions, local_names

    # -- per-chunk decision dump ------------------------------------------------------
    def dump_actions(self, name, metric, actions, local_names):
        if not self._dump_dir or (self._dump_pols and name not in self._dump_pols):
            return
        d = Path(self._dump_dir)
        d.mkdir(parents=True, exist_ok=True)
        with open(d / f'{self.wl}__{self.ph}__{metric}__{name}.csv', 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['chunk', 'config'])
            for i, a in enumerate(actions):
                w.writerow([i, local_names[a]])

    # -- the Viterbi action-path cache ------------------------------------------------
    def _viterbi_path(self, name, fn, args, metric, uses_model):
        """Cached full-trace DP path.

        The cache key carries the environment tag (traces + power mode + transition model) and,
        for model-based Viterbi, the prediction-set tag, because those change the optimum. It
        deliberately does NOT carry warmup: the DP is warmup-unaware, so the path is identical
        either way and only the re-scoring differs. Caching the trace instead of the path made
        the cache useless for every warmup run, which is every run in the chapter.
        """
        tag = self.env_tag if not uses_model else f'{self.env_tag}-{self.pred_tag}'
        cache_file = (self.viterbi_cache_dir / f"{self.wl}__{self.ph}__{name}__{tag}__{metric}.npz"
                      if self.viterbi_cache_dir is not None else None)
        if cache_file is not None and cache_file.exists():
            try:
                z = np.load(cache_file, allow_pickle=False)
                return z['actions'], [str(x) for x in z['names']]
            except Exception:
                pass                      # unreadable cache entry: recompute and overwrite
        _, actions, local_names = fn(*args, _return_actions=True, metric=metric)
        if cache_file is not None:
            np.savez(cache_file, actions=np.asarray(actions),
                     names=np.array(local_names, dtype='U16'))
        return actions, local_names

    # -- the dispatch ------------------------------------------------------------------
    def run(self, calls, traces_by_metric):
        """Evaluate every policy in `calls`, filling traces_by_metric[metric][name]."""
        for name, (fn, args) in calls.items():
            t0 = time.perf_counter() if self._time_pol else 0.0

            if getattr(fn, 'metric_independent', False):
                # One action sequence, scored under both metrics. Warmup must still be charged:
                # the heterogeneous heuristics are metric-independent AND they migrate, so
                # skipping it here would score them with no warmup while model policies pay.
                traces, actions, local_names = fn(*args, metrics=self.metrics, _return_actions=True)
                stats = compute_trace_stats(list(actions), local_names)
                for m in self.metrics:
                    self.dump_actions(name, m, actions, local_names)
                    traces_by_metric[m][name] = (self.trace_from_path(actions, local_names, m)
                                                 if self.apply_warmup else traces[m])
                    record_diag(self.diag_results, self.wl, self.ph, m, name, stats)

            elif getattr(fn, 'is_viterbi_oracle', False):
                if os.environ.get('SKIP_VITERBI'):
                    continue              # fast ladder: skip the expensive full-trace DPs
                uses_model = getattr(fn, 'uses_model', False)
                for m in self.metrics:
                    actions, local_names = self._viterbi_path(name, fn, args, m, uses_model)
                    stats = compute_trace_stats(list(actions), local_names)
                    self.dump_actions(name, m, actions, local_names)
                    traces_by_metric[m][name] = self.trace_from_path(actions, local_names, m)
                    record_diag(self.diag_results, self.wl, self.ph, m, name, stats)

            elif self.apply_warmup and getattr(fn, 'returns_actions', False):
                for m in self.metrics:
                    tr, stats, actions, local_names = self.apply_warmup_trace(fn, args, m)
                    self.dump_actions(name, m, actions, local_names)
                    traces_by_metric[m][name] = tr
                    record_diag(self.diag_results, self.wl, self.ph, m, name, stats)

            else:
                # Verbatim path: no warmup charge, no diagnostics row unless the policy
                # happens to report actions. See the module docstring.
                for m in self.metrics:
                    acts_out = []
                    tr, stats = call_with_stats(fn, *args, metric=m, _acts_out=acts_out)
                    if acts_out:
                        self.dump_actions(name, m, *acts_out[0])
                    traces_by_metric[m][name] = tr
                    record_diag(self.diag_results, self.wl, self.ph, m, name, stats)

            if self._time_pol:
                dt = time.perf_counter() - t0
                if dt > 1.0:
                    print(f"[SLOW POLICY] {self.wl} ph{self.ph}: {name} took {dt:.1f}s", flush=True)

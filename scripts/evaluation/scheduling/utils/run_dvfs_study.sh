#!/bin/bash
# run_dvfs_study.sh -- DVFS-only sweep (x86 P-core + E-core).
#
# The DVFS study depends only on the CROSS-FREQUENCY model, which is unaffected by
# the cross-processor training-data defect. It is therefore runnable and trustworthy
# right now, ahead of the cross-proc retrain.
#
# No cross_proc dirs are passed, so cross_proc_time_mat stays None and every
# heterogeneous MODEL policy is skipped by construction. Heuristics and oracles
# still run (they use no model) and are cheap. This makes the run substantially
# faster than the full sweep and keeps the output free of any cross-proc result.
#
#   ./run_dvfs_study.sh                    # raw model
#   ACC="raw cap20 cap10 cap5" ./run_dvfs_study.sh
#   DRY=1 ./run_dvfs_study.sh
#   BENCH_LIMIT=6 ./run_dvfs_study.sh      # smoke test on a few phases
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCHED_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$SCHED_DIR/../../.." && pwd)"
PY="$REPO_ROOT/.venv/bin/python3"
RES="$REPO_ROOT/results/scheduling"

TRACES="${TRACES:-$RES/Hetero_precompute/speedup_full_v2_repaired/granular_phase_traces}"
OUT_ROOT="${OUT_ROOT:-$RES/DVFS_Study}"
VITERBI="${VITERBI:-$RES/viterbi_cache_v2_persample}"
ACC="${ACC:-raw}"
DRY="${DRY:-0}"
POWER_MODE="${POWER_MODE:-per_sample}"
DECISION_POWER_MODE="${DECISION_POWER_MODE:-oracle}"
BENCH_LIMIT="${BENCH_LIMIT:-0}"
# Workloads with incomplete cross-frequency prediction sets. Excluded so the rest of the
# run keeps --strict_predictions on. Without this the loader would silently substitute
# oracle times for them, which makes the model policies read as near-perfect results.
# Re-dump these and clear the list.
# Previously excluded dacapo_cassandra,h2o,kafka,tradebeans,tradesoap: the cross-frequency
# translator lookup pointed at dacapo_c1_pruned (17 of 22 benches), so the dump skipped them
# and the loader silently substituted oracle times, which read as a perfect forecast result.
# Fixed (create_dataset._bench_suite_dir_cross_freq -> dacapo_c1) and re-dumped; all 22 now
# have predictions, so nothing is excluded. --strict_predictions is the backstop.
# 772.marian (90 chunks) and 706.stockfish (117 chunks) are the two shortest SPEC2026 workloads by
# nearly three orders of magnitude (the next shortest has 81,579 chunks). A walk-forward forecaster
# needs a training history longer than either trace holds, and the persistence gate needs ~50 chunks
# of realized error before it can engage, so neither policy is defined on them in any meaningful way.
# They are excluded from the evaluation. This is a property of the traces, not the method, and is
# noted in the chapter (sec:ch5_scope). Both cores drop them, so the exclusion is symmetric.
# Comma-separated: main.py --exclude_workloads splits on ',' (not whitespace).
EXCLUDE="${EXCLUDE:-spec_772.marian_r,spec_706.stockfish_r}"
# Forecast x Oracle bound. Set to the self_forecast dump root to enable
# Forecast_Oracle_{P,E,Hetero}; leave empty to skip those policies.
FC_ORACLE_DIR="${FC_ORACLE_DIR:-$RES/forecast_oracle_10M}"
[ -d "$FC_ORACLE_DIR" ] || FC_ORACLE_DIR=""
# Phase-UNAWARE forecasting arm (dumped with --method global). Same policy, same loaders;
# the forecaster just never segments by runtime phase. Paired with the phase-aware arm this
# measures whether phase-awareness converts into scheduling gains, rather than assuming the
# standalone MAPE result carries over. Leave empty to skip.
FC_UNAWARE_DIR="${FC_UNAWARE_DIR:-$RES/forecast_unaware_10M}"
[ -d "$FC_UNAWARE_DIR" ] || FC_UNAWARE_DIR=""
FC_ORACLE_UNAWARE_DIR="${FC_ORACLE_UNAWARE_DIR:-$RES/forecast_oracle_unaware_10M}"
[ -d "$FC_ORACLE_UNAWARE_DIR" ] || FC_ORACLE_UNAWARE_DIR=""
# Persistence-GATED forecasting arm. Falls back to the lagged persistence forecast whenever
# the model has been losing to it over a trailing window, decided per chunk from realized
# error only. Standalone MAPE (self-forecast, P@4GHz, 8.7M chunks) puts it at 4.87% against
# persistence 5.99% and the ungated model 6.05%, so it is the only forecasting arm that beats
# the reactive baseline on every suite. That makes it the arm the chapter's forecasting claim
# rests on, and it therefore has to be IN the study rather than measured only offline.
FC_GATED_DIR="${FC_GATED_DIR:-$RES/forecast_gated_10M}"
[ -d "$FC_GATED_DIR" ] || FC_GATED_DIR=""
FC_ORACLE_GATED_DIR="${FC_ORACLE_GATED_DIR:-$RES/forecast_oracle_gated_10M}"
[ -d "$FC_ORACLE_GATED_DIR" ] || FC_ORACLE_GATED_DIR=""

pred_dirs() {  # cross-frequency translate + forecast only. No cross-proc.
  local a=$1
  if [ "$a" = raw ]; then
    # forecast_oracle is a bound (forecast with perfect translation), not a model, so it
    # is fed only to the raw run: an error cap is defined against the cross-platform
    # model, and this policy has no cross-platform model to cap.
    echo "--cross_freq_p_pred_dir $RES/DVFS_precompute/cross_freq_translate_10M \
          --cross_freq_e_pred_dir $RES/DVFS_precompute/cross_freq_translate_10M \
          --cross_freq_p_forecast_dir $RES/DVFS_precompute/forecast_predictions_10M \
          --cross_freq_e_forecast_dir $RES/DVFS_precompute/forecast_predictions_10M \
          ${FC_ORACLE_DIR:+--forecast_oracle_dir $FC_ORACLE_DIR} \
          ${FC_UNAWARE_DIR:+--cross_freq_p_forecast_unaware_dir $FC_UNAWARE_DIR} \
          ${FC_UNAWARE_DIR:+--cross_freq_e_forecast_unaware_dir $FC_UNAWARE_DIR} \
          ${FC_ORACLE_UNAWARE_DIR:+--forecast_oracle_unaware_dir $FC_ORACLE_UNAWARE_DIR} \
          ${FC_GATED_DIR:+--cross_freq_p_forecast_gated_dir $FC_GATED_DIR} \
          ${FC_GATED_DIR:+--cross_freq_e_forecast_gated_dir $FC_GATED_DIR} \
          ${FC_ORACLE_GATED_DIR:+--forecast_oracle_gated_dir $FC_ORACLE_GATED_DIR}"
  else
    local n=${a#cap}
    echo "--cross_freq_p_pred_dir $RES/DVFS_precompute/capped/cf_tr_cap$n \
          --cross_freq_e_pred_dir $RES/DVFS_precompute/capped/cf_tr_cap$n \
          --cross_freq_p_forecast_dir $RES/DVFS_precompute/capped/cf_fc_cap$n \
          --cross_freq_e_forecast_dir $RES/DVFS_precompute/capped/cf_fc_cap$n"
  fi
}

fail=0
[ -x "$PY" ] || { echo "FATAL: no python at $PY"; fail=1; }
[ -d "$TRACES" ] || { echo "FATAL: traces not found: $TRACES"; fail=1; }
for a in $ACC; do
  for d in $(pred_dirs "$a" | tr ' ' '\n' | grep '^/'); do
    [ -d "$d" ] || { echo "FATAL: [$a] missing prediction dir: $d"; fail=1; }
  done
done
[ "$fail" = 0 ] || { echo; echo "Preflight failed. Nothing run."; exit 1; }

mkdir -p "$OUT_ROOT" "$VITERBI"
LIMIT_ARG=""
[ "$BENCH_LIMIT" != 0 ] && LIMIT_ARG="--limit $BENCH_LIMIT"
EXCL_ARG=""
[ -n "$EXCLUDE" ] && EXCL_ARG="--exclude_workloads $EXCLUDE"

echo "=== DVFS-only study (no cross-processor model)"
echo "    traces : $TRACES ($(ls "$TRACES" | wc -l) files)"
echo "    out    : $OUT_ROOT"
echo "    accs   : $ACC"
[ -n "$LIMIT_ARG" ] && echo "    LIMITED to $BENCH_LIMIT phases (smoke test)"
[ -n "$EXCLUDE" ] && echo "    EXCLUDED (incomplete predictions): $EXCLUDE"
[ -n "$FC_UNAWARE_DIR" ] && echo "    phase-UNAWARE arm: enabled" || echo "    phase-UNAWARE arm: skipped (dump missing)"
[ -n "$FC_GATED_DIR" ] && echo "    persistence-GATED arm: enabled" || echo "    persistence-GATED arm: skipped (dump missing)"
echo

for a in $ACC; do
  out="$OUT_ROOT/dvfs_$a"; log="$OUT_ROOT/dvfs_$a.log"
  cmd="$PY $SCHED_DIR/src/main.py \
    --input_dir $TRACES --output_dir $out \
    --power_mode $POWER_MODE --decision_power_mode $DECISION_POWER_MODE --apply_warmup --strict_predictions \
    --viterbi_cache_dir $VITERBI $LIMIT_ARG $EXCL_ARG $(pred_dirs "$a")"
  if [ "$DRY" = 1 ]; then echo "[$a] $cmd" | tr -s ' '; continue; fi
  mkdir -p "$out"   # after the DRY check, so a dry run leaves no empty dirs behind
  echo "[$a] starting -> $log"
  t0=$SECONDS
  if $cmd > "$log" 2>&1; then
    echo "[$a] done in $(( (SECONDS-t0)/60 ))m"
    grep -A6 "PREDICTION FALLBACKS" "$log" && echo "[$a] WARNING: fallbacks occurred"
  else
    echo "[$a] FAILED -- see $log"; tail -20 "$log"; exit 1
  fi
done

[ "$DRY" = 1 ] && exit 0
echo
echo "=== DVFS sweep complete. Check the ladder with:"
for a in $ACC; do
  echo "  $PY $SCRIPT_DIR/dvfs_report.py $OUT_ROOT/dvfs_$a"
done

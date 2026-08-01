#!/usr/bin/env bash
# Golden-output regression test for the simulator.
#
# The simulator produces the dissertation's numbers, so any refactor must be provably
# behaviour-preserving. This runs a small fixed workload set and compares every produced value
# against a stored golden copy, byte for byte.
#
#   utils/regression_test.sh record   capture the golden output from the current code
#   utils/regression_test.sh check    re-run and diff against it
#
# The workload set is deliberately tiny (two ~40-chunk SPEC workloads plus one DaCapo phase) so
# the test runs in a couple of minutes. Size does not matter here -- every policy, both metrics,
# the transition matrices, the warmup model and the Viterbi oracle are exercised regardless of
# trace length, and those are what a refactor can break.
#
# Uses its own Viterbi cache so it can never poison or depend on the real one.
#
# Rows are compared as a SET keyed by (Workload, Phase, Metric, Policy), not byte for byte:
# main.py collects results via concurrent.futures.as_completed, so CSV row order follows worker
# completion and varies between runs. A byte comparison reports every run as a mismatch.
set -e
cd "$(dirname "$0")/.."

PY=../../../.venv/bin/python3
RES=../../../results/scheduling
HP=$RES/Hetero_precompute
DP=$RES/DVFS_precompute
TRACES=$HP/speedup_full_v2_repaired/granular_phase_traces
GOLDEN=utils/golden
OUT=/tmp/regression_sim_$$
CACHE=/tmp/regression_viterbi_$$

KEEP="spec_772.marian_r spec_706.stockfish_r dacapo_avrora"
ALL=$(ls $TRACES | grep -oE 'spec_[0-9]+\.[a-z0-9]+_r|dacapo_[a-z0-9]+' | sort -u)
EXCL=$(comm -23 <(printf '%s\n' $ALL) <(printf '%s\n' $KEEP | tr ' ' '\n' | sort -u) | paste -sd,)

run () {
  mkdir -p "$CACHE"
  # SIM_EXPLORATORY=1 so the golden covers every policy, including the ones a production run gates off.
  FAST_HETERO=1 SIM_WORKERS=4 SIM_EXPLORATORY=1 $PY src/main.py --input_dir $TRACES --output_dir "$1" \
    --power_mode per_sample --decision_power_mode static --warmup_in_decision --apply_warmup \
    --strict_predictions --viterbi_cache_dir "$CACHE" --exclude_workloads "$EXCL" \
    --cross_freq_p_pred_dir $DP/cross_freq_translate_10M \
    --cross_freq_e_pred_dir $DP/cross_freq_translate_10M \
    --cross_freq_p_forecast_dir $DP/forecast_predictions_10M \
    --cross_freq_e_forecast_dir $DP/forecast_predictions_10M \
    --cross_proc_pred_dir $HP/cross_proc_translate_gentemporal_10M \
    --cross_proc_forecast_dir $HP/cross_proc_forecast_gentemporal_10M \
    > "$1/run.log" 2>&1
}

case "${1:-check}" in
  record)
    rm -rf $GOLDEN; mkdir -p $GOLDEN $OUT
    run "$OUT"
    cp "$OUT/all_phases_summary.csv" $GOLDEN/
    [ -f "$OUT/diagnostics.csv" ] && cp "$OUT/diagnostics.csv" $GOLDEN/
    rm -rf "$OUT" "$CACHE"
    echo "recorded golden output:"
    wc -l $GOLDEN/*.csv
    ;;
  check)
    [ -d $GOLDEN ] || { echo "no golden output; run '$0 record' first" >&2; exit 1; }
    mkdir -p "$OUT"
    run "$OUT"
    fail=0
    for f in $GOLDEN/*.csv; do
      b=$(basename "$f")
      if ! $PY utils/_cmp_golden.py "$f" "$OUT/$b"; then
        fail=1
      fi
    done
    rm -rf "$OUT" "$CACHE"
    [ $fail -eq 0 ] && echo "PASS: every value matches golden (row order ignored)" || {
      echo "FAIL: refactor changed simulator output"; exit 1; }
    ;;
  *) echo "usage: $0 {record|check}" >&2; exit 1 ;;
esac

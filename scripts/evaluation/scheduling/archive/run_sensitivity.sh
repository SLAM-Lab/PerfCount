#!/usr/bin/env bash
# Two sensitivity sweeps on the deployable (general) cross-processor model.
#
#   A. CAPPED ERROR  - clamp the general model's predictions to within +/-N% of ground truth,
#                      isolating translation accuracy from the scheduling policy.
#                      caps 0/5/10/20 plus the raw model already in hetero/gentemporal.
#
#   B. MIGRATION COST - scale the realized migration cost upward. The measured defaults are mild
#                      against a 10M-instruction chunk (warmup decays inside a single chunk,
#                      context switch is ~0.2% of one), so this asks whether the heterogeneous
#                      conclusions survive a machine where migration genuinely hurts.
#                        mid  : 2x warmup depth, 3x warmup duration, 10x context switch
#                        high : 3x warmup depth, 6x warmup duration, 50x context switch
#
# Usage: ./run_sensitivity.sh [capped|migcost|all]
set -euo pipefail
cd "$(dirname "$0")"
PY=../../../.venv/bin/python3
RES=${RES:-../../../results/scheduling}
HP=$RES/Hetero_precompute
DP=$RES/DVFS_precompute
TRACES=$HP/speedup_full_v2_repaired/granular_phase_traces
SUITES="spec_2017 spec_2026 dacapo_c1"
NBENCH=69
WHICH=${1:-all}

have () { ls "$1"/speedups_from_P_1.0GHz 2>/dev/null | sed -E 's/_phase[0-9]+\.csv//' | sort -u | wc -l; }

# sim <outdir> <cp_translate_dir> <cp_forecast_dir>
sim () {
  FAST_HETERO=1 $PY src/main.py --input_dir $TRACES --output_dir $RES/sensitivity/$1 \
    --power_mode per_sample --decision_power_mode static --warmup_in_decision --apply_warmup \
    --strict_predictions --viterbi_cache_dir $HP/viterbi_cache_hetero \
    --cross_freq_p_pred_dir $DP/cross_freq_translate_10M \
    --cross_freq_e_pred_dir $DP/cross_freq_translate_10M \
    --cross_freq_p_forecast_dir $DP/forecast_predictions_10M \
    --cross_freq_e_forecast_dir $DP/forecast_predictions_10M \
    --cross_proc_pred_dir "$2" --cross_proc_forecast_dir "$3"
}

# ---------------- A. capped error ----------------
if [ "$WHICH" = capped ] || [ "$WHICH" = all ]; then
  echo "=== capped-error sweep (general model, caps 5/10/20) ==="
  GT=$HP/cross_proc_translate_gentemporal_10M
  GF=$HP/cross_proc_forecast_gentemporal_10M
  for c in 5 10 20; do
    [ "$(have $HP/capped/gen_tr_cap$c)" -ge $NBENCH ] || \
      $PY utils/cap_predictions.py --pred_dir $GT --granular $TRACES --out_base $HP/capped/gen_tr --caps $c
    [ "$(have $HP/capped/gen_fc_cap$c)" -ge $NBENCH ] || \
      $PY utils/cap_predictions.py --pred_dir $GF --granular $TRACES --out_base $HP/capped/gen_fc --caps $c
  done
  for c in 5 10 20; do
    echo "--- cap$c ---"
    sim cap$c $HP/capped/gen_tr_cap$c $HP/capped/gen_fc_cap$c
  done
fi

# ---------------- B. migration cost ----------------
if [ "$WHICH" = migcost ] || [ "$WHICH" = all ]; then
  echo "=== migration-cost sweep (general model) ==="
  echo "--- migcost_mid (A x2, tau x3, ctxsw x10) ---"
  WARMUP_A_SCALE=2 WARMUP_TAU_SCALE=3 MIG_LAT_SCALE=10 \
    sim migcost_mid $HP/cross_proc_translate_gentemporal_10M $HP/cross_proc_forecast_gentemporal_10M
  echo "--- migcost_high (A x3, tau x6, ctxsw x50) ---"
  WARMUP_A_SCALE=3 WARMUP_TAU_SCALE=6 MIG_LAT_SCALE=50 \
    sim migcost_high $HP/cross_proc_translate_gentemporal_10M $HP/cross_proc_forecast_gentemporal_10M
fi

echo "=== done: $RES/sensitivity ==="

#!/usr/bin/env bash
# NEED-4 / NEED-5 / NEED-6: the three sim studies unblocked by the power-ratio and hybrid
# prediction generators. Every prediction directory this needs already exists, so unlike
# run_hetero.sh / run_dvfs.sh there is no build phase here -- this is simulation only.
#
#   A. hetero, predicted per-sample decision power   (NEED-5)  -> hetero/predpower_<model>
#   B. DVFS,   predicted per-sample decision power   (NEED-4)  -> DVFS/predpower_<model>
#   C. hetero, hybrid crossover-corrected preds      (NEED-6)  -> hetero/hybrid
#
# The power predictions cover SPEC only (120 of 147 phases). DaCapo has no measured power to
# fit a ratio against, and the two ultra-short SPEC2026 workloads are excluded elsewhere too.
# Phases without a prediction file fall back to the characterized static table, never to truth
# (src/main.py:344). So the DaCapo rows in the predpower studies are identical to the static
# baseline by construction and must not be read as a power result.
set -e
cd "$(dirname "$0")"

PY=../../../.venv/bin/python3
RES=../../../results/scheduling
PMU=../../../processed_data_10M/x86_desktop_heterogeneous
HP=$RES/Hetero_precompute
DP=$RES/DVFS_precompute
TRACES=$HP/speedup_full_v2_repaired/granular_phase_traces
VITERBI=$HP/viterbi_cache_hetero
PWR=$HP/power_pred_10M

# Share the box with the forecasting sweep instead of fighting it.
export SIM_WORKERS=${SIM_WORKERS:-40}

SUITES="spec_2017 spec_2026 dacapo_c1"
BENCHES=$( { ls $TRACES | grep -oE 'spec_[0-9]+\.[a-z0-9]+_r' | sort -u
             find $PMU/dacapo_c1 -name 'aligned_dacapo_*_cpu0_phase*.csv' -printf '%f\n' 2>/dev/null \
               | grep -oE 'dacapo_[a-z0-9]+' | sort -u; } | sort -u | paste -sd' ')
EXCL=$( comm -23 <(ls $TRACES | grep -oE 'spec_[0-9]+\.[a-z0-9]+_r|dacapo_[a-z0-9]+' | sort -u) \
                 <(printf '%s\n' $BENCHES | sort -u) | paste -sd, )
NBENCH=$(echo $BENCHES | wc -w)

# Fail before burning hours if a prediction set is short, rather than silently simulating a
# partial study.
need () { n=$(ls "$1/speedups_from_P_1.0GHz/" 2>/dev/null | grep -oE 'spec_[0-9]+\.[a-z0-9]+_r|dacapo_[a-z0-9]+' | sort -u | wc -l)
          [ "$n" -ge "$NBENCH" ] || { echo "MISSING: $1 covers $n/$NBENCH workloads" >&2; exit 1; } }
need $HP/cross_proc_translate_10M;             need $HP/cross_proc_forecast_10M
need $HP/cross_proc_translate_gentemporal_10M; need $HP/cross_proc_forecast_gentemporal_10M
need $HP/cross_proc_translate_geninsample_10M; need $HP/cross_proc_forecast_geninsample_10M
need $HP/capped/cp_tr_cap0;                    need $HP/capped/cp_fc_cap0
need $HP/cross_proc_translate_hybrid_10M;      need $HP/cross_proc_forecast_hybrid_10M
need $DP/cross_freq_translate_10M;             need $DP/forecast_predictions_10M
need $DP/cross_freq_translate_gentemporal_10M; need $DP/cross_freq_forecast_gentemporal_10M
need $DP/cross_freq_translate_geninsample_10M; need $DP/cross_freq_forecast_geninsample_10M
need $DP/capped/cf_tr_cap0;                    need $DP/capped/cf_fc_cap0
[ -d "$PWR" ] || { echo "MISSING: $PWR" >&2; exit 1; }
echo "=== $NBENCH workloads, SIM_WORKERS=$SIM_WORKERS, power preds: $(ls $PWR | wc -l) phases ==="

# $1=out  $2=cross_proc_translate  $3=cross_proc_forecast  $4... = extra flags
sim_het () {
  local out=$1 tr=$2 fc=$3; shift 3
  FAST_HETERO=1 $PY src/main.py --input_dir $TRACES --output_dir $RES/hetero/$out \
    --power_mode per_sample --warmup_in_decision --apply_warmup \
    --strict_predictions --viterbi_cache_dir $VITERBI ${EXCL:+--exclude_workloads $EXCL} \
    --cross_freq_p_pred_dir $DP/cross_freq_translate_10M --cross_freq_e_pred_dir $DP/cross_freq_translate_10M \
    --cross_freq_p_forecast_dir $DP/forecast_predictions_10M --cross_freq_e_forecast_dir $DP/forecast_predictions_10M \
    --cross_proc_pred_dir $tr --cross_proc_forecast_dir $fc "$@"
}

# $1=out  $2=cross_freq_translate  $3=cross_freq_forecast  $4... = extra flags
sim_dvfs () {
  local out=$1 tr=$2 fc=$3; shift 3
  FAST_HETERO=1 $PY src/main.py --input_dir $TRACES --output_dir $RES/DVFS/$out \
    --power_mode per_sample --apply_warmup \
    --strict_predictions --viterbi_cache_dir $VITERBI ${EXCL:+--exclude_workloads $EXCL} \
    --cross_proc_pred_dir $HP/cross_proc_translate_10M --cross_proc_forecast_dir $HP/cross_proc_forecast_10M \
    --cross_freq_p_pred_dir $tr --cross_freq_e_pred_dir $tr \
    --cross_freq_p_forecast_dir $fc --cross_freq_e_forecast_dir $fc "$@"
}

PRED="--decision_power_mode predicted --decision_power_dir $PWR"

# --- A. NEED-5: heterogeneous with predicted per-sample decision power ---------------
echo "=== A. hetero + predicted power ==="
sim_het predpower_loocv       $HP/cross_proc_translate_10M             $HP/cross_proc_forecast_10M             $PRED
sim_het predpower_gentemporal $HP/cross_proc_translate_gentemporal_10M $HP/cross_proc_forecast_gentemporal_10M $PRED
sim_het predpower_geninsample $HP/cross_proc_translate_geninsample_10M $HP/cross_proc_forecast_geninsample_10M $PRED
sim_het predpower_oracle      $HP/capped/cp_tr_cap0                    $HP/capped/cp_fc_cap0                   $PRED

# --- B. NEED-4: DVFS with predicted per-sample decision power -------------------------
echo "=== B. DVFS + predicted power ==="
sim_dvfs predpower_loocv       $DP/cross_freq_translate_10M             $DP/forecast_predictions_10M             $PRED
sim_dvfs predpower_gentemporal $DP/cross_freq_translate_gentemporal_10M $DP/cross_freq_forecast_gentemporal_10M  $PRED
sim_dvfs predpower_geninsample $DP/cross_freq_translate_geninsample_10M $DP/cross_freq_forecast_geninsample_10M  $PRED
sim_dvfs predpower_oracle      $DP/capped/cf_tr_cap0                    $DP/capped/cf_fc_cap0                    $PRED

# --- C. NEED-6: heterogeneous with hybrid crossover-corrected predictions -------------
# Static decision power, so this isolates the classifier correction from the power effect.
echo "=== C. hetero + hybrid predictions ==="
sim_het hybrid $HP/cross_proc_translate_hybrid_10M $HP/cross_proc_forecast_hybrid_10M --decision_power_mode static

# --- compare -------------------------------------------------------------------------
echo; echo "=== A. hetero: predicted power vs static baseline ==="
RES=$RES $PY compare_runs.py hetero loocv predpower_loocv gentemporal predpower_gentemporal \
    geninsample predpower_geninsample oracle predpower_oracle
echo; echo "=== B. DVFS: predicted power vs static baseline ==="
RES=$RES $PY compare_runs.py dvfs loocv predpower_loocv gentemporal predpower_gentemporal \
    geninsample predpower_geninsample oracle predpower_oracle
echo; echo "=== C. hetero: hybrid vs cross-proc alone (loocv) ==="
RES=$RES $PY compare_runs.py hetero loocv hybrid

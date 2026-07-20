#!/usr/bin/env bash
# Heterogeneous scheduling across 4 cross-processor models: loocv, gen-temporal,
# gen-insample, oracle. FULLY SELF-HEALING: builds every prediction it needs, regenerating
# anything that does not yet cover the full workload set, then runs the simulator and prints
# a reactive / forecast / gate table. Just run it: ./run_hetero.sh  (first run takes hours).
set -e
cd "$(dirname "$0")"                                          # scripts/evaluation/scheduling

PY=../../../.venv/bin/python3
RES=../../../results/scheduling
PMU=../../../processed_data_10M/x86_desktop_heterogeneous
CPMODELS=../../../results/cross_platform/cross_proc/x86_10M
CFMODELS=../../../results/cross_platform/cross_freq/x86_10M
DUMP=../workload_forecasting/phase_forecasting/run_dump_dvfs.sh
HP=$RES/Hetero_precompute        # cross-processor predictions (built here)
DP=$RES/DVFS_precompute          # cross-frequency predictions (the fixed "other half")
TRACES=$HP/speedup_full_v2_repaired/granular_phase_traces
VITERBI=$HP/viterbi_cache_hetero

# suites + workloads: SPEC2017 + SPEC2026 + DaCapo c1 (c1/c2 are different counter cohorts).
SUITES="spec_2017 spec_2026 dacapo_c1"
BENCHES=$( { ls $TRACES | grep -oE 'spec_[0-9]+\.[a-z0-9]+_r' | sort -u
             find $PMU/dacapo_c1 -name 'aligned_dacapo_*_cpu0_phase*.csv' -printf '%f\n' 2>/dev/null \
               | grep -oE 'dacapo_[a-z0-9]+' | sort -u; } | sort -u | paste -sd' ')
EXCL=$( comm -23 <(ls $TRACES | grep -oE 'spec_[0-9]+\.[a-z0-9]+_r|dacapo_[a-z0-9]+' | sort -u) \
                 <(printf '%s\n' $BENCHES | sort -u) | paste -sd, )
NBENCH=$(echo $BENCHES | wc -w)

# distinct workloads already present in a prediction dir, and the ones still missing
have () { ls "$1/speedups_from_P_1.0GHz/" 2>/dev/null | grep -oE 'spec_[0-9]+\.[a-z0-9]+_r|dacapo_[a-z0-9]+' | sort -u | wc -l; }
miss () { comm -23 <(printf '%s\n' $BENCHES | sort -u) \
                   <(ls "$1/speedups_from_P_1.0GHz/" 2>/dev/null | grep -oE 'spec_[0-9]+\.[a-z0-9]+_r|dacapo_[a-z0-9]+' | sort -u) | paste -sd' '; }

# build cross-PROCESSOR translate+forecast for one model (only if it lacks some workload).
# $1=feature_set  $2=translate_dir  $3=forecast_dir
build_cp () {
  [ "$(have $2)" -ge $NBENCH ] || $PY cross_proc_precompute.py --model_dir $CPMODELS --pmu_dir $PMU \
      --oracle_dir $TRACES --out_dir $2 --feature_set $1 --suites $SUITES
  [ "$(have $3)" -ge $NBENCH ] || CBM_FEATURE_SET=$1 XPROC=1 BENCHES="$(miss $3)" CORES="0 16" \
      FREQS="1.0 2.0 3.0 4.0" OUT_DIR=$3 bash $DUMP
}

# the fixed cross-FREQUENCY half (LOOCV top4) that every hetero run holds constant
build_cf_fixed () {
  if [ "$(have $DP/cross_freq_translate_10M)" -lt $NBENCH ]; then
    for c in P E; do $PY cross_freq_precompute.py --model_base_dir $CFMODELS --pmu_dir $PMU \
        --oracle_dir $TRACES --out_dir $DP/cross_freq_translate_10M --core_type $c --feature_set top4 --suites $SUITES; done
  fi
  [ "$(have $DP/forecast_predictions_10M)" -ge $NBENCH ] || CBM_FEATURE_SET=top4 XPROC=0 \
      BENCHES="$(miss $DP/forecast_predictions_10M)" CORES="0 16" FREQS="1.0 2.0 3.0 4.0" \
      OUT_DIR=$DP/forecast_predictions_10M bash $DUMP
}

# run the simulator for one model.  $1=name  $2=cross_proc_translate  $3=cross_proc_forecast
sim () {
  FAST_HETERO=1 $PY src/main.py --input_dir $TRACES --output_dir $RES/hetero/$1 \
    --power_mode per_sample --decision_power_mode static --warmup_in_decision --apply_warmup \
    --strict_predictions --viterbi_cache_dir $VITERBI ${EXCL:+--exclude_workloads $EXCL} \
    --cross_freq_p_pred_dir $DP/cross_freq_translate_10M --cross_freq_e_pred_dir $DP/cross_freq_translate_10M \
    --cross_freq_p_forecast_dir $DP/forecast_predictions_10M --cross_freq_e_forecast_dir $DP/forecast_predictions_10M \
    --cross_proc_pred_dir $2 --cross_proc_forecast_dir $3
}

echo "=== run_hetero: $NBENCH workloads ($SUITES) ==="

# --- build every prediction set (skips whatever already covers all workloads) ----------
build_cf_fixed                                                                            # fixed cross-freq half
build_cp top4             $HP/cross_proc_translate_10M             $HP/cross_proc_forecast_10M
build_cp general_temporal $HP/cross_proc_translate_gentemporal_10M $HP/cross_proc_forecast_gentemporal_10M
build_cp general_insample $HP/cross_proc_translate_geninsample_10M $HP/cross_proc_forecast_geninsample_10M
# oracle = clamp the loocv predictions to ground truth (cap0)
[ "$(have $HP/capped/cp_tr_cap0)" -ge $NBENCH ] || $PY utils/cap_predictions.py --pred_dir $HP/cross_proc_translate_10M --granular $TRACES --out_base $HP/capped/cp_tr --caps 0
[ "$(have $HP/capped/cp_fc_cap0)" -ge $NBENCH ] || $PY utils/cap_predictions.py --pred_dir $HP/cross_proc_forecast_10M  --granular $TRACES --out_base $HP/capped/cp_fc --caps 0

# --- run each model ---------------------------------------------------------
sim loocv       $HP/cross_proc_translate_10M             $HP/cross_proc_forecast_10M
sim gentemporal $HP/cross_proc_translate_gentemporal_10M $HP/cross_proc_forecast_gentemporal_10M
sim geninsample $HP/cross_proc_translate_geninsample_10M $HP/cross_proc_forecast_geninsample_10M
sim oracle      $HP/capped/cp_tr_cap0                    $HP/capped/cp_fc_cap0

# --- compare ----------------------------------------------------------------
echo; echo "=== reactive / forecast / gate  (policy / oracle, lower=better) ==="
RES=$RES $PY compare_runs.py hetero loocv gentemporal geninsample oracle

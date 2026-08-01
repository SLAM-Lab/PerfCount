#!/usr/bin/env bash
# Reproduce every heterogeneous-scheduling and DVFS number in Chapter 5, at both model
# qualities, from one launch. Idempotent: an arm whose summary already exists is skipped
# unless FORCE=1. Prints the gap to the oracle at the end through report_chapter5.py.
#
#   ./run_chapter5.sh              run whatever is missing, then report
#   FORCE=1 ./run_chapter5.sh      re-run every arm from scratch (repeatability check)
#   ./run_chapter5.sh report       only print the report from existing results
#   DEPLOYABLE=1 ./run_chapter5.sh run the general and loocv arms with the incumbent
#                                  configuration scored from the PREVIOUS sample rather than the
#                                  true next one (what a real scheduler has), into *_deployable
#                                  dirs, then report the idealized-vs-deployable difference
#
# What it runs, all with the warmup-correct gate and gentemporal cross-frequency throughout:
#   heterogeneous, cross-processor model quality axis:
#       loocv      per-workload leave-one-out cross-processor model
#       general    single general model trained across workloads (the deployable operating point)
#       perfectcp  cross-processor predictions clamped to ground truth (the model bound)
#   DVFS, cross-frequency model quality axis:
#       loocv, general
# Every run also contains the heuristics, the reactive-oracle, the perfect-future greedy, the
# greedy policy with TRUE per-sample power, and the global Viterbi oracle, so the full ladder
# and its gap decomposition come out of the same summaries.
#
# The two prior framework fixes this depends on are in the code, not here: the reactive-fallback
# gate now declares returns_actions so it pays the cross-cluster warmup penalty (POLICIES.md),
# and the measured platform constants live in src/platform_model.py.
set -e
cd "$(dirname "$0")"

PY=../../../.venv/bin/python3
RES=../../../results/scheduling
PMU=../../../processed_data_10M/x86_desktop_heterogeneous
HP=$RES/Hetero_precompute
DP=$RES/DVFS_precompute
TRACES=$HP/speedup_full_v2_repaired/granular_phase_traces
VITERBI=$HP/viterbi_cache_hetero
export SIM_WORKERS=${SIM_WORKERS:-40}
export OMP_NUM_THREADS=1

# Deployable-diagonal mode. When on, the incumbent configuration is scored from the previous
# sample rather than the true next one, which is what a scheduler actually has. Results go to
# *_deployable dirs so they sit beside the idealized runs for comparison. See data_loader
# DEPLOYABLE_DIAGONAL and POLICIES.md.
if [ -n "$DEPLOYABLE" ]; then
  export DEPLOYABLE_DIAGONAL=1
  SUF=_deployable
else
  export DEPLOYABLE_DIAGONAL=0
  SUF=
fi

# Case-insensitive on the workload name so cactuBSSN and the like are not silently dropped.
BENCHES=$( { ls $TRACES | grep -oE 'spec_[0-9]+\.[A-Za-z0-9]+_r' | sort -u
             find $PMU/dacapo_c1 -name 'aligned_dacapo_*_cpu0_phase*.csv' -printf '%f\n' 2>/dev/null \
               | grep -oE 'dacapo_[A-Za-z0-9]+' | sort -u; } | sort -u | paste -sd' ')
EXCL=$( comm -23 <(ls $TRACES | grep -oE 'spec_[0-9]+\.[A-Za-z0-9]+_r|dacapo_[A-Za-z0-9]+' | sort -u) \
                 <(printf '%s\n' $BENCHES | sort -u) | paste -sd, )
NB=$(echo $BENCHES | wc -w)

# Fixed halves, held constant across each axis so a comparison isolates the varying model.
HET_CF_TR=$DP/cross_freq_translate_gentemporal_10M     # heterogeneous holds cross-freq at general
HET_CF_FC=$DP/cross_freq_forecast_gentemporal_10M
DVFS_CP_TR=$HP/cross_proc_translate_gentemporal_10M    # DVFS holds cross-proc at general
DVFS_CP_FC=$HP/cross_proc_forecast_gentemporal_10M

need () {  # fail early rather than simulate a partial prediction set
  local n; n=$(ls "$1/speedups_from_P_1.0GHz/" 2>/dev/null \
      | grep -oE 'spec_[0-9]+\.[A-Za-z0-9]+_r|dacapo_[A-Za-z0-9]+' | sort -u | wc -l)
  [ "$n" -ge "$NB" ] || { echo "MISSING prediction set ($n/$NB): $1" >&2
      echo "build it with run_hetero.sh / run_dvfs.sh first" >&2; exit 1; }
}

done_already () { [ -z "$FORCE" ] && [ -f "$RES/$1/all_phases_summary.csv" ]; }

sim_het () {  # $1=out-name  $2=cross_proc_translate  $3=cross_proc_forecast
  local out="$1$SUF"
  if done_already "hetero/$out"; then echo "  hetero/$out present, skipping (FORCE=1 to redo)"; return; fi
  need "$2"; need "$3"
  echo "=== hetero/$out ${DEPLOYABLE:+(deployable diagonal)} ==="
  FAST_HETERO=1 $PY src/main.py --input_dir $TRACES --output_dir $RES/hetero/$out \
    --power_mode per_sample --decision_power_mode static --warmup_in_decision --apply_warmup \
    --strict_predictions --viterbi_cache_dir $VITERBI ${EXCL:+--exclude_workloads $EXCL} \
    --cross_freq_p_pred_dir $HET_CF_TR --cross_freq_e_pred_dir $HET_CF_TR \
    --cross_freq_p_forecast_dir $HET_CF_FC --cross_freq_e_forecast_dir $HET_CF_FC \
    --cross_proc_pred_dir $2 --cross_proc_forecast_dir $3
}

sim_dvfs () {  # $1=out-name  $2=cross_freq_translate  $3=cross_freq_forecast
  local out="$1$SUF"
  if done_already "DVFS/$out"; then echo "  DVFS/$out present, skipping (FORCE=1 to redo)"; return; fi
  need "$2"; need "$3"
  echo "=== DVFS/$out ${DEPLOYABLE:+(deployable diagonal)} ==="
  FAST_HETERO=1 $PY src/main.py --input_dir $TRACES --output_dir $RES/DVFS/$out \
    --power_mode per_sample --decision_power_mode static --apply_warmup \
    --strict_predictions --viterbi_cache_dir $VITERBI ${EXCL:+--exclude_workloads $EXCL} \
    --cross_proc_pred_dir $DVFS_CP_TR --cross_proc_forecast_dir $DVFS_CP_FC \
    --cross_freq_p_pred_dir $2 --cross_freq_e_pred_dir $2 \
    --cross_freq_p_forecast_dir $3 --cross_freq_e_forecast_dir $3
}

if [ "${1:-run}" != "report" ]; then
  echo "=== $NB workloads, SIM_WORKERS=$SIM_WORKERS ${FORCE:+(FORCE: re-running all arms)}${DEPLOYABLE:+ (DEPLOYABLE diagonal)} ==="
  # Heterogeneous: vary the cross-processor model
  sim_het general   $HP/cross_proc_translate_gentemporal_10M $HP/cross_proc_forecast_gentemporal_10M
  sim_het loocv     $HP/cross_proc_translate_10M             $HP/cross_proc_forecast_10M
  # perfectcp: the cross-processor bound. Under DEPLOYABLE it keeps the honest diagonal too, so it
  # bounds the deployable operating point consistently rather than mixing diagonals in one table.
  sim_het perfectcp $HP/capped/cp_tr_cap0 $HP/capped/cp_fc_cap0
  # DVFS: vary the cross-frequency model
  sim_dvfs general  $DP/cross_freq_translate_gentemporal_10M $DP/cross_freq_forecast_gentemporal_10M
  sim_dvfs loocv    $DP/cross_freq_translate_10M             $DP/forecast_predictions_10M
fi

echo; echo "=== REPORT ==="
if [ -n "$DEPLOYABLE" ]; then
  RES=$RES $PY report_chapter5.py --deployable
else
  RES=$RES $PY report_chapter5.py
fi

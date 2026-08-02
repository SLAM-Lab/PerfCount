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
#       loocv, general, then general clamped to +-20% / +-10% / 0% error (cap20, cap10, cap0),
#       the model-accuracy sweep behind fig:dvfs_forecast_curve
# Each axis is run both at static decision power (deployable, default) and at true per-sample
# power (*_true dirs), the latter feeding the comparison tables, the planning figure, the
# per-workload figures, and the forecast-curve figure.
# Every run also contains the heuristics, the reactive-oracle, the perfect-future greedy, the
# greedy policy with TRUE per-sample power, and the global Viterbi oracle, so the full ladder
# and its gap decomposition come out of the same summaries.
#
# PIPELINE. This script is the middle stage and does not build the prediction models:
#   1. UPSTREAM (run first)   run_hetero.sh + run_dvfs.sh build the cross-processor and
#                             cross-frequency prediction tensors (loocv + general), the Viterbi
#                             cache, the granular phase traces, and the perfectcp cap0 dirs.
#                             This script asserts they exist (the need/have_cap guards) and stops
#                             with a build hint if any are missing, so it never simulates a
#                             partial prediction set. The DVFS cap20/cap10/cap0 dirs it builds
#                             itself from the general cross-frequency model.
#   2. THIS SCRIPT            runs every sim arm above and prints the report (report_chapter5.py).
#   3. DOWNSTREAM (figures)   plotting_scripts/chapter_5/*.py read the summaries produced here and
#                             render the PDFs (e.g. fig04_dvfs_forecast_curve.py). This script
#                             writes summaries and the text report only, not figures.
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

sim_het () {  # $1=out-name  $2=cross_proc_translate  $3=cross_proc_forecast  [$4=decision_power: static|oracle]
  local dpm="${4:-static}"; local out="$1$SUF"
  [ "$dpm" = oracle ] && out="${1}_true$SUF"   # true per-sample decision power -> *_true dirs
  if done_already "hetero/$out"; then echo "  hetero/$out present, skipping (FORCE=1 to redo)"; return; fi
  need "$2"; need "$3"
  echo "=== hetero/$out (decision power: $dpm) ${DEPLOYABLE:+(deployable diagonal)} ==="
  FAST_HETERO=1 $PY src/main.py --input_dir $TRACES --output_dir $RES/hetero/$out \
    --power_mode per_sample --decision_power_mode $dpm --warmup_in_decision --apply_warmup \
    --strict_predictions --viterbi_cache_dir $VITERBI ${EXCL:+--exclude_workloads $EXCL} \
    --cross_freq_p_pred_dir $HET_CF_TR --cross_freq_e_pred_dir $HET_CF_TR \
    --cross_freq_p_forecast_dir $HET_CF_FC --cross_freq_e_forecast_dir $HET_CF_FC \
    --cross_proc_pred_dir $2 --cross_proc_forecast_dir $3
}

sim_dvfs () {  # $1=out-name  $2=cross_freq_translate  $3=cross_freq_forecast  [$4=decision_power: static|oracle]
  local dpm="${4:-static}"; local out="$1$SUF"
  [ "$dpm" = oracle ] && out="${1}_true$SUF"
  if done_already "DVFS/$out"; then echo "  DVFS/$out present, skipping (FORCE=1 to redo)"; return; fi
  need "$2"; need "$3"
  echo "=== DVFS/$out (decision power: $dpm) ${DEPLOYABLE:+(deployable diagonal)} ==="
  FAST_HETERO=1 $PY src/main.py --input_dir $TRACES --output_dir $RES/DVFS/$out \
    --power_mode per_sample --decision_power_mode $dpm --apply_warmup \
    --strict_predictions --viterbi_cache_dir $VITERBI ${EXCL:+--exclude_workloads $EXCL} \
    --cross_proc_pred_dir $DVFS_CP_TR --cross_proc_forecast_dir $DVFS_CP_FC \
    --cross_freq_p_pred_dir $2 --cross_freq_e_pred_dir $2 \
    --cross_freq_p_forecast_dir $3 --cross_freq_e_forecast_dir $3
}

if [ "${1:-run}" != "report" ]; then
  echo "=== $NB workloads, SIM_WORKERS=$SIM_WORKERS ${FORCE:+(FORCE: re-running all arms)}${DEPLOYABLE:+ (DEPLOYABLE diagonal)} ==="
  # have_cap: count workloads covered by a capped prediction dir, for the idempotent build guards.
  have_cap () { ls "$1/speedups_from_P_1.0GHz/" 2>/dev/null \
      | grep -oE 'spec_[0-9]+\.[A-Za-z0-9]+_r|dacapo_[A-Za-z0-9]+' | sort -u | wc -l; }
  # Heterogeneous: vary the cross-processor model (static-decision-power deployable arms)
  sim_het general   $HP/cross_proc_translate_gentemporal_10M $HP/cross_proc_forecast_gentemporal_10M
  sim_het loocv     $HP/cross_proc_translate_10M             $HP/cross_proc_forecast_10M
  # perfectcp: the cross-processor bound, cross-processor predictions clamped to ground truth.
  # Built here from the loocv cross-processor model, mirroring run_hetero.sh; cap0 clamps to truth
  # so the base is immaterial. Under DEPLOYABLE it keeps the honest diagonal too, so it bounds the
  # deployable operating point consistently rather than mixing diagonals in one table.
  [ "$(have_cap $HP/capped/cp_tr_cap0)" -ge "$NB" ] || \
    $PY utils/cap_predictions.py --pred_dir $HP/cross_proc_translate_10M --granular $TRACES --out_base $HP/capped/cp_tr --caps 0
  [ "$(have_cap $HP/capped/cp_fc_cap0)" -ge "$NB" ] || \
    $PY utils/cap_predictions.py --pred_dir $HP/cross_proc_forecast_10M  --granular $TRACES --out_base $HP/capped/cp_fc --caps 0
  sim_het perfectcp $HP/capped/cp_tr_cap0 $HP/capped/cp_fc_cap0
  # DVFS: vary the cross-frequency model (static-decision-power deployable arms)
  sim_dvfs general  $DP/cross_freq_translate_gentemporal_10M $DP/cross_freq_forecast_gentemporal_10M
  sim_dvfs loocv    $DP/cross_freq_translate_10M             $DP/forecast_predictions_10M

  # True per-sample decision-power arms (-> *_true dirs). These supply the "true per-sample power"
  # columns of the comparison tables, the planning figure, and the per-workload figures, so those
  # numbers come from this script rather than ad-hoc pwrcmp/DVFS_Study runs.
  sim_het  general $HP/cross_proc_translate_gentemporal_10M $HP/cross_proc_forecast_gentemporal_10M oracle
  sim_dvfs general $DP/cross_freq_translate_gentemporal_10M $DP/cross_freq_forecast_gentemporal_10M oracle
  sim_dvfs loocv   $DP/cross_freq_translate_10M             $DP/forecast_predictions_10M             oracle

  # Cross-frequency error-cap sweep for the model-accuracy figure (fig:dvfs_forecast_curve).
  # Clamp the GENERAL cross-frequency model's predictions to within +-N% of ground truth, which
  # isolates translation accuracy from the policy. cap0 is the perfect model. The base is the
  # general model so the axis reads loocv -> general -> caps -> perfect, and the arms run under
  # true per-sample power so the whole curve is power-consistent with the loocv_true/general_true
  # endpoints it sits between.
  CF_GT=$DP/cross_freq_translate_gentemporal_10M
  CF_GF=$DP/cross_freq_forecast_gentemporal_10M
  for c in 20 10 0; do
    [ "$(have_cap $DP/capped/cf_gen_tr_cap$c)" -ge "$NB" ] || \
      $PY utils/cap_predictions.py --pred_dir $CF_GT --granular $TRACES --out_base $DP/capped/cf_gen_tr --caps $c
    [ "$(have_cap $DP/capped/cf_gen_fc_cap$c)" -ge "$NB" ] || \
      $PY utils/cap_predictions.py --pred_dir $CF_GF --granular $TRACES --out_base $DP/capped/cf_gen_fc --caps $c
  done
  sim_dvfs cap20 $DP/capped/cf_gen_tr_cap20 $DP/capped/cf_gen_fc_cap20 oracle
  sim_dvfs cap10 $DP/capped/cf_gen_tr_cap10 $DP/capped/cf_gen_fc_cap10 oracle
  sim_dvfs cap0  $DP/capped/cf_gen_tr_cap0  $DP/capped/cf_gen_fc_cap0  oracle
fi

echo; echo "=== REPORT ==="
if [ -n "$DEPLOYABLE" ]; then
  RES=$RES $PY report_chapter5.py --deployable
else
  RES=$RES $PY report_chapter5.py
fi

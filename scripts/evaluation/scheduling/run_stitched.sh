#!/usr/bin/env bash
# TRUE heterogeneous-history forecasting.
#
# The shipped forecast tensors assume the workload never moved: for each source configuration,
# that config's whole trace is translated to every target and forecast from. The simulator then
# reads the slice matching wherever the policy currently sits, so after a migration the forecast
# is conditioned on a history that never happened in that run.
#
# This builds the history the workload actually had -- raw measurements where it was already on
# the target, translations of wherever it actually was otherwise -- and forecasts from that.
#
# CONTROLLED COMPARISON. Only the FORECAST tensors change. The reactive/translate half, the
# policies, the power mode and the warmup accounting are identical to the baseline
# (results/scheduling/hetero/gatefix), so any difference is the history, not the setup.
#
# Both halves use the general_temporal translator. run_hetero.sh historically pinned the
# cross-frequency half to top4 while varying only cross-processor, but general_temporal carries
# far more decision-relevant signal (ratio-domain correlation 0.586 against top4's 0.158), so
# there is no reason to hobble one axis. Using one variant everywhere also means the stitcher
# can emit both halves in a single pass -- CBM_FEATURE_SET is a global read at import, so mixed
# variants would otherwise require one invocation per half.
#
# The BASELINE must use the same pair of translators or the comparison is confounded. That run
# is results/scheduling/hetero/gt_baseline, not hetero/gatefix (which pins cross-freq to top4).
#
# The path is policy- and metric-specific, so the tensors are too. Read each metric's rows from
# its own simulator run.
set -e
cd "$(dirname "$0")"

# CatBoost and BLAS each spawn thread pools. With many stitch processes in parallel that
# oversubscribes the machine and segfaults (seen on dacapo_sunflow). run_dump_dvfs.sh pins these
# for the same reason: parallelism here comes from the process fan-out, not from threads.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 TF_CPP_MIN_LOG_LEVEL=3

PY=../../../.venv/bin/python3
RES=../../../results/scheduling
PMU=../../../processed_data_10M/x86_desktop_heterogeneous
HP=$RES/Hetero_precompute
DP=$RES/DVFS_precompute
TRACES=$HP/speedup_full_v2_repaired/granular_phase_traces
VITERBI=$HP/viterbi_cache_hetero
STITCH=../workload_forecasting/phase_forecasting/stitch_history.py

POLICY=${POLICY:-Model_Forecast_ReactiveGated_Hetero}
METRIC=${METRIC:-EDP}
ACTS=$RES/actions_stitch_EDP          # dumped by the gatefix run; holds both metrics
OUT_PROC=$HP/cross_proc_forecast_stitched_${METRIC}_10M
OUT_FREQ=$DP/cross_freq_forecast_stitched_${METRIC}_10M
PAR=${PAR:-24}
export SIM_WORKERS=${SIM_WORKERS:-32}

BENCHES=$( { ls $TRACES | grep -oE 'spec_[0-9]+\.[A-Za-z0-9]+_r' | sort -u
             find $PMU/dacapo_c1 -name 'aligned_dacapo_*_cpu0_phase*.csv' -printf '%f\n' 2>/dev/null \
               | grep -oE 'dacapo_[A-Za-z0-9]+' | sort -u; } | sort -u | paste -sd' ')
EXCL=$( comm -23 <(ls $TRACES | grep -oE 'spec_[0-9]+\.[A-Za-z0-9]+_r|dacapo_[A-Za-z0-9]+' | sort -u) \
                 <(printf '%s\n' $BENCHES | sort -u) | paste -sd, )
NB=$(echo $BENCHES | wc -w)

[ -d "$ACTS" ] && [ -n "$(ls -A $ACTS 2>/dev/null)" ] || {
  echo "no action dumps in $ACTS -- run the simulator with DUMP_ACTIONS_DIR first" >&2; exit 1; }

run_one () {
  $PY "$STITCH" --bench "$1" --actions_dir "$ACTS" --policy "$POLICY" --metric "$METRIC" \
      --out_proc "$OUT_PROC" --out_freq "$OUT_FREQ" --emit "$EMIT" --feature_set "$FS" \
    >/dev/null 2>&1 && echo "  ok: $1" || echo "  FAIL: $1"
}
export -f run_one
export PY STITCH ACTS POLICY METRIC OUT_PROC OUT_FREQ

# $1 = half (proc|freq), $2 = translator variant, $3 = that half's output dir
stitch_half () {
  local have
  have=$(ls "$OUT_PROC/speedups_from_P_1.0GHz/" 2>/dev/null \
          | grep -oE 'spec_[0-9]+\.[A-Za-z0-9]+_r|dacapo_[A-Za-z0-9]+' | sort -u | wc -l)
  if [ "$have" -ge "$NB" ]; then echo "=== $1 half already complete ($have/$NB) ==="; return; fi
  echo "=== stitching $1 half with the $2 translator ($NB benches) ==="
  export CBM_FEATURE_SET=$2 EMIT=$1 FS=$2
  printf '%s\n' $BENCHES | xargs -P "$PAR" -I{} bash -c 'run_one {}'
}

mkdir -p $OUT_PROC $OUT_FREQ
stitch_half both general_temporal $OUT_PROC     # one variant everywhere -> one pass

echo "=== simulating on the stitched forecasts ==="
FAST_HETERO=1 $PY src/main.py --input_dir $TRACES --output_dir $RES/hetero/stitched_${METRIC} \
  --power_mode per_sample --decision_power_mode static --warmup_in_decision --apply_warmup \
  --strict_predictions --viterbi_cache_dir $VITERBI ${EXCL:+--exclude_workloads $EXCL} \
  --cross_freq_p_pred_dir $DP/cross_freq_translate_gentemporal_10M \
  --cross_freq_e_pred_dir $DP/cross_freq_translate_gentemporal_10M \
  --cross_freq_p_forecast_dir $OUT_FREQ --cross_freq_e_forecast_dir $OUT_FREQ \
  --cross_proc_pred_dir $HP/cross_proc_translate_gentemporal_10M \
  --cross_proc_forecast_dir $OUT_PROC
echo "=== done -> $RES/hetero/stitched_${METRIC}  (compare against hetero/gatefix) ==="

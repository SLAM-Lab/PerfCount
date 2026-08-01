#!/usr/bin/env bash
# Match the forecast horizon to the commitment window.
#
# The forecast tensors are horizon=1: row i predicts chunk i. The commit policies hold a
# configuration for W chunks. So Model_Forecast_CommitW currently commits for ten chunks to
# whichever configuration looked best for the *next one*, and it is catastrophic (-1% to -7%
# against reactive). Horizon and window are mismatched.
#
# It also explains why the plain forecast barely beats reactive. Reactive IS persistence, and
# at h=1 persistence is near-optimal -- the chapter's own numbers have persistence at 9.2/14.5%
# MAPE against the forecaster oracle's 13.9%. Over a W-chunk window that reverses: carrying one
# value forward for ten chunks is a poor estimate of the window, while a window-mean forecast
# averages away the per-chunk noise. dump_dvfs_forecast --horizon K predicts the mean of the
# next K chunks for exactly this reason.
#
# Ceiling for this direction, from MPC_Oracle_W5/W10 in the corrected run (windowed planning
# with perfect information): +1.12% EDP and +2.66% ED2P on SPEC2017, against the plain gate's
# +0.71% / +0.31%.
#
# Feature sets follow run_hetero.sh exactly: cross-PROC varies (general_temporal), cross-FREQ is
# the fixed top4 half. Getting these wrong silently compares against a different translator.
set -e
cd "$(dirname "$0")"

PY=../../../.venv/bin/python3
RES=../../../results/scheduling
PMU=../../../processed_data_10M/x86_desktop_heterogeneous
HP=$RES/Hetero_precompute
DP=$RES/DVFS_precompute
TRACES=$HP/speedup_full_v2_repaired/granular_phase_traces
VITERBI=$HP/viterbi_cache_hetero
DUMP=../workload_forecasting/phase_forecasting/run_dump_dvfs.sh
PAR=${PAR:-40}
export SIM_WORKERS=${SIM_WORKERS:-40}

SUITES="spec_2017 spec_2026 dacapo_c1"
BENCHES=$( { ls $TRACES | grep -oE 'spec_[0-9]+\.[A-Za-z0-9]+_r' | sort -u
             find $PMU/dacapo_c1 -name 'aligned_dacapo_*_cpu0_phase*.csv' -printf '%f\n' 2>/dev/null \
               | grep -oE 'dacapo_[A-Za-z0-9]+' | sort -u; } | sort -u | paste -sd' ')
EXCL=$( comm -23 <(ls $TRACES | grep -oE 'spec_[0-9]+\.[A-Za-z0-9]+_r|dacapo_[A-Za-z0-9]+' | sort -u) \
                 <(printf '%s\n' $BENCHES | sort -u) | paste -sd, )
NB=$(echo $BENCHES | wc -w)

have () { ls "$1/speedups_from_P_1.0GHz/" 2>/dev/null | grep -oE 'spec_[0-9]+\.[A-Za-z0-9]+_r|dacapo_[A-Za-z0-9]+' | sort -u | wc -l; }
miss () { comm -23 <(printf '%s\n' $BENCHES | sort -u) \
                   <(ls "$1/speedups_from_P_1.0GHz/" 2>/dev/null | grep -oE 'spec_[0-9]+\.[A-Za-z0-9]+_r|dacapo_[A-Za-z0-9]+' | sort -u) | paste -sd' '; }

for H in ${HORIZONS:-5 10}; do
  # Both halves use the gentemporal translator, matching run_chapter5.sh's general arm, so the
  # Commit-K result is comparable to the chapter's gate. The _gt tag on the cross-frequency dir
  # keeps it distinct from any earlier top4 horizon dump so this re-dumps cleanly rather than
  # reusing a mismatched one.
  PROC=$HP/cross_proc_forecast_h${H}_10M
  FREQ=$DP/cross_freq_forecast_h${H}_gt_10M
  echo "=== horizon $H: dumping ($NB benches) ==="
  if [ "$(have $PROC)" -lt "$NB" ]; then
    CBM_FEATURE_SET=general_temporal XPROC=1 HORIZON=$H PAR=$PAR \
      BENCHES="$(miss $PROC)" CORES="0 16" FREQS="1.0 2.0 3.0 4.0" OUT_DIR=$PROC bash $DUMP
  fi
  if [ "$(have $FREQ)" -lt "$NB" ]; then
    CBM_FEATURE_SET=general_temporal XPROC=0 HORIZON=$H PAR=$PAR \
      BENCHES="$(miss $FREQ)" CORES="0 16" FREQS="1.0 2.0 3.0 4.0" OUT_DIR=$FREQ bash $DUMP
  fi
  [ "$(have $PROC)" -ge "$NB" ] || { echo "horizon $H proc dump incomplete: $(have $PROC)/$NB" >&2; exit 1; }
  [ "$(have $FREQ)" -ge "$NB" ] || { echo "horizon $H freq dump incomplete: $(have $FREQ)/$NB" >&2; exit 1; }

  echo "=== horizon $H: simulating (read Model_Forecast_Commit${H}_Hetero from this run) ==="
  FAST_HETERO=1 $PY src/main.py --input_dir $TRACES --output_dir $RES/hetero/h${H} \
    --power_mode per_sample --decision_power_mode static --warmup_in_decision --apply_warmup \
    --strict_predictions --viterbi_cache_dir $VITERBI ${EXCL:+--exclude_workloads $EXCL} \
    --cross_freq_p_pred_dir $DP/cross_freq_translate_gentemporal_10M --cross_freq_e_pred_dir $DP/cross_freq_translate_gentemporal_10M \
    --cross_freq_p_forecast_dir $FREQ --cross_freq_e_forecast_dir $FREQ \
    --cross_proc_pred_dir $HP/cross_proc_translate_gentemporal_10M --cross_proc_forecast_dir $PROC
  echo "=== horizon $H done -> $RES/hetero/h${H} ==="
done
echo "=== ALL HORIZONS COMPLETE ==="

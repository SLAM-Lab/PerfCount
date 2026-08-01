#!/usr/bin/env bash
# Phase-UNAWARE forecast arm, for the aware-vs-unaware comparison inside the simulator.
#
# The chapter's phase-aware claim currently rests on standalone forecast accuracy (MAPE). The
# scheduler question is different: does phase-awareness change a DECISION? That needs the unaware
# forecasts dumped with the same models and feature sets, differing only in --method, so the
# simulator can score Model_Forecast_Unaware_* against Model_Forecast_*.
#
# METHOD=global is the phase-unaware forecaster (one model for the whole workload);
# the aware arm uses the default per_phase.
set -euo pipefail
cd "$(dirname "$0")"
PY=../../../.venv/bin/python3
RES=${RES:-../../../results/scheduling}
PMU=../../../processed_data_10M/x86_desktop_heterogeneous
HP=$RES/Hetero_precompute
DP=$RES/DVFS_precompute
TRACES=$HP/speedup_full_v2_repaired/granular_phase_traces
DUMP=../workload_forecasting/phase_forecasting/run_dump_dvfs.sh
PAR=${PAR:-40}
BENCHES=$( { ls $TRACES | grep -oE 'spec_[0-9]+\.[a-z0-9]+_r' | sort -u
             ls $TRACES | grep -oE 'dacapo_[a-z0-9]+'          | sort -u; } | paste -sd' ' )
echo "=== phase-unaware dump: $(echo $BENCHES | wc -w) benches, PAR=$PAR ==="

# cross-FREQUENCY unaware (matches forecast_predictions_10M, top4)
if [ ! -d "$DP/forecast_predictions_unaware_10M/speedups_from_P_1.0GHz" ]; then
  echo "--- cross-freq unaware ---"
  CBM_FEATURE_SET=top4 XPROC=0 METHOD=global PAR=$PAR BENCHES="$BENCHES" \
    CORES="0 16" FREQS="1.0 2.0 3.0 4.0" OUT_DIR=$DP/forecast_predictions_unaware_10M bash $DUMP
fi

# cross-PROCESSOR unaware (matches cross_proc_forecast_gentemporal_10M, general_temporal)
if [ ! -d "$HP/cross_proc_forecast_unaware_gentemporal_10M/speedups_from_P_1.0GHz" ]; then
  echo "--- cross-proc unaware (general_temporal) ---"
  CBM_FEATURE_SET=general_temporal XPROC=1 METHOD=global PAR=$PAR BENCHES="$BENCHES" \
    CORES="0 16" FREQS="1.0 2.0 3.0 4.0" OUT_DIR=$HP/cross_proc_forecast_unaware_gentemporal_10M bash $DUMP
fi
echo "=== unaware dumps done ==="

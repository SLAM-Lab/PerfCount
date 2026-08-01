#!/bin/bash
# run_phase_aware_ch4.sh
# ======================
# Regenerates the phase-aware forecasting results behind Chapter 4's
# "Phase-Aware Forecasting" paragraph and Figure PerfCount_Figure_13_Phase_Aware.
#
# Adds no logic of its own: it calls the existing run_all_models.sh once per core
# with the settings the chapter describes.
#
#   full workload set   (not run_all_models.sh's balanced 24-benchmark default)
#   GMM, 6 phases, 10M granularity, horizon 1, five-window history
#   DELTA=1  -> residual-over-persistence targets, as the chapter reports
#   SAVE=1   -> persists per-phase ensembles, which the inference-latency
#               experiment (latency/measure_phase_latency.py) needs
#
# Writes results/forecasting/phase_forecasting/phase_forecast_10M_<model>_gmm_delta_cpu{0,16}.csv
# -- exactly the files plotting_scripts/forecasting/phase_aware/plot_phase_aware.py reads.
#
# The box is heavily oversubscribed while the regen sweep runs, so by default this
# WAITS for a given PID to exit before starting. Set WAIT_PID=0 to start now.
#
# Usage:
#   WAIT_PID=1388085 ./run_phase_aware_ch4.sh      # queue behind the regen driver
#   WAIT_PID=0 PAR=20 ./run_phase_aware_ch4.sh     # start immediately, gently
#   MODELS="dt mlp" ./run_phase_aware_ch4.sh       # subset
#   DRY=1 ./run_phase_aware_ch4.sh                 # print, run nothing

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
DATA="$REPO/processed_data_10M/x86_desktop_heterogeneous"
LOGD="$REPO/results/forecasting/phase_forecasting"

FREQ="${FREQ:-4.0}"
MODELS="${MODELS:-dt mlp lstm transformer}"
PAR="${PAR:-40}"
WAIT_PID="${WAIT_PID:-0}"
DRY="${DRY:-0}"

# The workload set is derived from the traces present for BOTH cores at $FREQ, so
# the two cores are evaluated on the same benchmarks and the set is not hardcoded.
list_for_cpu() {
  ls "$DATA"/spec_2017/spec_2017_4ghz "$DATA"/spec_2026/spec_2026_4ghz \
     "$DATA"/dacapo_c1/dacapo_c1_4ghz 2>/dev/null \
    | grep -oP "^aligned_\K.*(?=_${FREQ}GHz_cpu${1}_phase)" | sort -u
}
BENCHES="$(comm -12 <(list_for_cpu 0) <(list_for_cpu 16) | tr '\n' ' ')"
NB=$(echo $BENCHES | wc -w)

echo "=== Chapter 4 phase-aware sweep ==="
echo "  workloads : $NB (present on both cores at ${FREQ}GHz)"
echo "  models    : $MODELS"
echo "  parallel  : $PAR      delta: on   save-models: on"
echo "  outputs   : $LOGD/phase_forecast_10M_<model>_gmm_delta_cpu{0,16}.csv"
[ "$NB" -eq 0 ] && { echo "  no workloads found under $DATA -- aborting"; exit 1; }

if [ "$WAIT_PID" != 0 ]; then
  echo
  echo "  waiting for PID $WAIT_PID to exit before starting..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
  echo "  PID $WAIT_PID has exited; starting at $(date '+%H:%M:%S')"
fi

for CPU in 0 16; do
  echo
  printf '=%.0s' {1..72}; echo
  echo "CPU $CPU  ($([ "$CPU" = 0 ] && echo P-core || echo E-core))  start $(date '+%H:%M:%S')"
  printf '=%.0s' {1..72}; echo
  [ "$DRY" = 1 ] && { echo "+ MODELS=\"$MODELS\" CPU=$CPU DELTA=1 SAVE=1 PAR=$PAR BENCHES=<$NB> run_all_models.sh"; continue; }
  MODELS="$MODELS" CPU="$CPU" FREQ="$FREQ" DELTA=1 SAVE=1 PAR="$PAR" BENCHES="$BENCHES" \
    bash "$SCRIPT_DIR/run_all_models.sh"
done

echo
echo "Done $(date '+%H:%M:%S'). Produced:"
ls -l "$LOGD"/phase_forecast_10M_*_gmm_delta_cpu*.csv 2>/dev/null || echo "  (none -- check the logs above)"
echo "Next: python plotting_scripts/forecasting/phase_aware/plot_phase_aware.py"

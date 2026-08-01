#!/bin/bash
# regen_all_forecasting.sh
# ========================
# Thin orchestration: run the EXISTING forecasting launchers end to end, in
# dependency order, to confirm they work and the results reproduce. This script
# adds no logic of its own -- it only calls existing launchers/plotters.
#
#   1. Homogeneous baselines (Ch. 4): x86 P-core + E-core + Arm N1 top4 sweeps,
#      then the baseline-grid figures, per-workload tables, and imagick figure.
#   2/3. Heterogeneous training + inference + figures (Ch. 5): the existing
#      run_forecasting_under_heterogeneity.sh (which does train, infer, figs).
#
# The homogeneous top4 MODELS are the per-config forecasters the heterogeneous
# INFERENCE stage reuses, so stage 1 runs before stage 2.
#
# Usage:
#   ./regen_all_forecasting.sh              # run everything
#   DRY=1 ./regen_all_forecasting.sh        # print the commands, run nothing
#   MAX_WORKERS=40 ./regen_all_forecasting.sh
#
# NOTE: the existing launchers SKIP already-successful jobs. For a true
# from-scratch rerun, move the target outputs aside first, e.g.
#   cd results/forecasting
#   mkdir _old && mv logs_10M/*_top4 logs_10M/*_het_* models_10M/*_top4 \
#                    cross_config/*.csv _old/

set -uo pipefail

WF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"          # workload_forecasting/
REPO="$(cd "$WF/../../.." && pwd)"
PY="$REPO/.venv/bin/python3"
HOM="$WF/homogeneous_forecasting"
PLOT="$REPO/plotting_scripts/forecasting/homogeneous_baseline"
MAX_WORKERS="${MAX_WORKERS:-80}"

run() { echo "+ $*"; [ "${DRY:-0}" = 1 ] && return 0; "$@"; }
hr()  { printf '=%.0s' {1..72}; echo; }

hr; echo "1/3  HOMOGENEOUS BASELINES  (x86 P-core, E-core, Arm N1; top4)"; hr
run "$PY" "$HOM/x86_desktop/parallel_launcher_top4_sweep.py" --cpu 0  --max_workers "$MAX_WORKERS"
run "$PY" "$HOM/x86_desktop/parallel_launcher_top4_sweep.py" --cpu 16 --max_workers "$MAX_WORKERS"
run "$PY" "$HOM/arm_server/parallel_launcher_top4_sweep.py"
run "$PY" "$PLOT/plot_baseline_grid.py"
run "$PY" "$PLOT/gen_perworkload_table.py"
run "$PY" "$PLOT/plot_imagick_phases.py"

echo
hr; echo "2/3 + 3/3  HETEROGENEOUS TRAINING + INFERENCE + FIGURES"; hr
DRY="${DRY:-0}" MAX_WORKERS="$MAX_WORKERS" \
  bash "$WF/heterogeneous_forecasting/run_forecasting_under_heterogeneity.sh"

echo
hr; echo "Done. Regenerated Ch. 4 baseline figures + Ch. 5 heterogeneity results."; hr

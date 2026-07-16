#!/bin/bash
# run_forecasting.sh — MASTER orchestrator for the x86_desktop phase-aware forecasting study.
# ============================================================================================
# Launches the complete experiment matrix start-to-finish, all on the config-invariant top4b
# counter set {ref_cycles, cpu_cycles, branches, instructions}:
#
#   Stage 1  Homogeneous phase forecasting   (all models, absolute + delta)
#   Stage 2  Heterogeneous-history TRAINING  (all models, {cross_freq,cross_proc} x prob sweep)
#   Stage 3  Heterogeneous INFERENCE         (all models, both directions vs translated-persistence)
#   Stage 4  Analysis rollup                 (multi-phase split; het recovery; het-inference)
#
# Everything writes under results/forecasting/phase_forecasting/ and (SAVE=1, default) saves the
# trained ensembles under .../phase_forecasting/models/.  No forecaster pretraining is needed —
# the phase harness trains its own per-phase/global models on every run.
#
# Scope via env (defaults = thorough / all workloads):
#   MODELS="dt mlp lstm transformer"   PROBS="0.2 0.4 0.6 0.8 1.0"   DIRECTIONS="cross_freq cross_proc"
#   BENCHES="<subset>"  (default: ALL cpu$CPU @ ${FREQ}GHz benches)   PAR=80   NN_EPOCHS=30   SAVE=1
# Jobs are single-threaded; on this 160-core box PAR=80 is the default. Watch RAM on all-NN stages
# (~1.5-2 GB/job -> ~120-160 GB at PAR=80, of ~190 GB free).
#
# Examples:
#   ./run_forecasting.sh                              # full matrix, all benches, all models
#   MODELS=dt ./run_forecasting.sh                    # DT-only fast end-to-end pass
#   BENCHES="spec_505.mcf_r dacapo_h2" MODELS=dt PROBS=0.3 ./run_forecasting.sh   # quick smoke
#
# NOTE: the full matrix (NN models x prob sweep x directions x ~68 benches) is LARGE
# (hours-to-days). DT stages complete first for early signal; run from tmux. Scope via env
# for quick passes.

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

MODELS="${MODELS:-dt mlp lstm transformer}"
PROBS="${PROBS:-0.2 0.4 0.6 0.8 1.0}"
DIRECTIONS="${DIRECTIONS:-cross_freq cross_proc}"
COUNTERS="${COUNTERS:-ref_cycles cpu_cycles branches instructions}"   # top4b
PAR="${PAR:-80}"; NN_EPOCHS="${NN_EPOCHS:-30}"; SAVE="${SAVE:-1}"
FREQ="${FREQ:-4.0}"; CPU="${CPU:-0}"; DATASET="${DATASET:-x86_desktop_heterogeneous}"

# Discover ALL target benchmarks (cpu$CPU @ ${FREQ}GHz) unless the caller pins BENCHES.
if [ -z "${BENCHES:-}" ]; then
  DATA="$REPO_ROOT/processed_data_10M/$DATASET"
  BENCHES=$(find "$DATA" -name "aligned_*_${FREQ}GHz_cpu${CPU}_phase*.csv" 2>/dev/null \
    | sed -E "s#.*/aligned_(.+)_${FREQ}GHz_cpu${CPU}_phase[0-9]+\.csv#\1#" | sort -u | tr '\n' ' ')
fi
NB=$(echo $BENCHES | wc -w)

echo "############################################################################"
echo "# run_forecasting : x86_desktop phase-aware forecasting study (top4b)"
echo "#   benches   : $NB    models: [$MODELS]"
echo "#   het probs : [$PROBS]    directions: [$DIRECTIONS]"
echo "#   par: $PAR   nn_epochs: $NN_EPOCHS   save: $SAVE   target: cpu$CPU @ ${FREQ}GHz"
echo "############################################################################"; echo

export BENCHES COUNTERS PAR NN_EPOCHS SAVE FREQ CPU DATASET

# ---- Stage 1: homogeneous (absolute + delta) ----
echo ">>>>> STAGE 1: homogeneous phase forecasting (absolute)"
MODELS="$MODELS" DELTA=0 "$SCRIPT_DIR/run_all_models.sh" || true
echo ">>>>> STAGE 1: homogeneous phase forecasting (delta)"
MODELS="$MODELS" DELTA=1 "$SCRIPT_DIR/run_all_models.sh" || true

# ---- Stage 2: heterogeneous-history TRAINING (direction x prob x model) ----
for DIR in $DIRECTIONS; do
  for P in $PROBS; do
    for M in $MODELS; do
      echo ">>>>> STAGE 2: het-training  dir=$DIR prob=$P model=$M"
      MODE="$DIR" HET_PROB="$P" MODEL="$M" DELTA=1 "$SCRIPT_DIR/run_het_phase.sh" || true
    done
  done
done

# ---- Stage 3: heterogeneous INFERENCE (both directions per model, via SOURCES) ----
for M in $MODELS; do
  echo ">>>>> STAGE 3: het-inference model=$M (cross_freq + cross_proc)"
  MODEL="$M" DELTA=1 "$SCRIPT_DIR/run_het_inference.sh" || true
done

# ---- Stage 4: analysis rollup ----
echo; echo ">>>>> STAGE 4: analysis rollup"
RR="$REPO_ROOT/results/forecasting/phase_forecasting"
PY="$REPO_ROOT/.venv/bin/python3"
"$PY" "$SCRIPT_DIR/analyze_multiphase.py" 2>/dev/null || true
for f in "$RR"/het_phase_*.csv;  do [ -f "$f" ] && { echo "--- $(basename "$f") ---"; "$PY" "$SCRIPT_DIR/analyze_het_phase.py"     --csv "$f" || true; }; done
for f in "$RR"/het_infer_*.csv;  do [ -f "$f" ] && { echo "--- $(basename "$f") ---"; "$PY" "$SCRIPT_DIR/analyze_het_inference.py" --csv "$f" || true; }; done

echo; echo "############### run_forecasting complete -> $RR ###############"

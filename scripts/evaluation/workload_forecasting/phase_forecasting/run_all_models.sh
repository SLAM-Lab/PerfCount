#!/bin/bash
# run_all_models.sh
# =================
# Phase-level forecasting across ALL FOUR forecaster models (DT / MLP / LSTM /
# Transformer) on one config (cpu0 @ 4.0GHz), a balanced 24-benchmark set
# (8 DaCapo / 8 SPEC2017 / 8 SPEC2026), GMM phases. For every (model, benchmark)
# it evaluates: global, per_phase (route on current phase), per_phase_pred
# (route on a LEARNED next-phase prediction, Alcorta-style), per_phase_oracle_next
# (route on the TRUE next phase = upper bound), per_phase_gated (DT only), and
# persistence.
#
# Parallelized across (model, benchmark); each job writes its own CSV, then all are
# concatenated per model -> phase_forecast_10M_{model}_gmm.csv, and analyzed.
#
# Model-specific handling: LSTM forced --stateless (per-phase subsets have arbitrary
# size); NN models skip the (expensive) selection gate and use fewer epochs; DT keeps
# the gate. Override anything via env.
#
# Usage:
#   ./run_all_models.sh
#   MODELS="dt mlp" PAR=12 ./run_all_models.sh
#   CLASSIFIER=gmm PHASE_COUNT=8 NN_EPOCHS=50 ./run_all_models.sh
#   BENCHES="dacapo_avrora spec_505.mcf_r" ./run_all_models.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WF_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$WF_DIR/../../.." && pwd)"
PYTHON="$REPO_ROOT/.venv/bin/python3"
HARNESS="$SCRIPT_DIR/predict_phase_forecast.py"

export PYTHONPATH="$WF_DIR"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 \
       TF_CPP_MIN_LOG_LEVEL=3

MODELS="${MODELS:-dt mlp lstm transformer}"
DATASET="${DATASET:-x86_desktop_heterogeneous}"
CPU="${CPU:-0}"; FREQ="${FREQ:-4.0}"
CLASSIFIER="${CLASSIFIER:-gmm}"; PHASE_COUNT="${PHASE_COUNT:-6}"; TIMESTEPS="${TIMESTEPS:-5}"
COUNTERS="${COUNTERS:-ref_cycles cpu_cycles branches instructions}"   # top4b (config-invariant)
NN_EPOCHS="${NN_EPOCHS:-30}"    # NN training epochs (DT ignores)
PAR="${PAR:-80}"
# DELTA=1 -> residual-over-persistence targets (predict x[i+1]-x[i]); output tagged _delta.
DELTA_FLAG=""; DTAG=""; [ "${DELTA:-0}" = "1" ] && { DELTA_FLAG="--delta"; DTAG="_delta"; }
# SAVE=1 -> persist trained ensembles under results/.../phase_forecasting/models/.
SAVE_FLAG=""; [ "${SAVE:-0}" = "1" ] && SAVE_FLAG="--save_models_dir $REPO_ROOT/results/forecasting/phase_forecasting/models"

# Balanced 24-benchmark default (8 per suite).
BENCHES="${BENCHES:-\
dacapo_avrora dacapo_batik dacapo_h2 dacapo_jython dacapo_lusearch dacapo_pmd dacapo_sunflow dacapo_xalan \
spec_500.perlbench_r spec_505.mcf_r spec_508.namd_r spec_519.lbm_r spec_520.omnetpp_r spec_523.xalancbmk_r spec_525.x264_r spec_557.xz_r \
spec_707.ntest_r spec_708.sqlite_r spec_710.omnetpp_r spec_714.cpython_r spec_721.gcc_r spec_723.llvm_r spec_731.astcenc_r spec_777.zstd_r}"

RR="$REPO_ROOT/results/forecasting/phase_forecasting"
mkdir -p "$RR"
TMPD="$(mktemp -d)"; trap 'rm -rf "$TMPD"' EXIT

echo "=== Phase-level forecasting: all models (cpu$CPU @ ${FREQ}GHz, $CLASSIFIER/$PHASE_COUNT) ==="
echo "  models   : $MODELS"
echo "  benches  : $(echo $BENCHES | wc -w)   parallelism: $PAR   nn_epochs: $NN_EPOCHS"
echo

run_one() {
    local m="$1" b="$2"
    local wl="aligned_${b}_${FREQ}GHz_cpu${CPU}"
    local extra=""
    local mpt=64
    if [ "$m" != "dt" ]; then extra="--no_gate --epochs $NN_EPOCHS"; mpt=128; fi
    [ "$m" = "lstm" ] && extra="$extra --stateless"
    "$PYTHON" "$HARNESS" --benchmark "$wl" --dataset "$DATASET" \
        --input_counters $COUNTERS --model "$m" \
        --classifier "$CLASSIFIER" --phase_count "$PHASE_COUNT" \
        --timesteps "$TIMESTEPS" --min_phase_train "$mpt" $extra $DELTA_FLAG $SAVE_FLAG \
        --out "$TMPD/${m}__${b}.csv" >/dev/null 2>&1
    echo "  done: $m | $b"
}
export -f run_one
export PYTHON HARNESS DATASET FREQ CPU COUNTERS CLASSIFIER PHASE_COUNT TIMESTEPS NN_EPOCHS TMPD DELTA_FLAG SAVE_FLAG

# Dispatch all (model, bench) jobs in parallel.
JOBS="$TMPD/jobs.txt"; : > "$JOBS"
for m in $MODELS; do for b in $BENCHES; do echo "$m $b" >> "$JOBS"; done; done
cat "$JOBS" | xargs -P "$PAR" -L 1 bash -c 'run_one "$0" "$1"'

# Per model: concat this model's per-bench CSVs, then analyze.
for m in $MODELS; do
    OUT="$RR/phase_forecast_10M_${m}_${CLASSIFIER}${DTAG}_cpu${CPU}.csv"
    first=1; : > "$OUT"
    for b in $BENCHES; do
        f="$TMPD/${m}__${b}.csv"; [ -f "$f" ] || continue
        if [ "$first" = 1 ]; then cat "$f" > "$OUT"; first=0; else tail -n +2 "$f" >> "$OUT"; fi
    done
    echo; echo "########## model = $m  ->  $OUT ##########"
    "$PYTHON" "$SCRIPT_DIR/analyze_phase_forecast.py" --csv "$OUT" || true
done

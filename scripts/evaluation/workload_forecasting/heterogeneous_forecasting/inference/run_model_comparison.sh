#!/bin/bash
# run_model_comparison.sh
# =======================
# Cross-config model comparison at HORIZON 1: for every benchmark, run all four
# forecasters (DT / MLP / LSTM / Transformer) over the full source->target config
# grid, so we can see which model best predicts the next timestep given a history
# unified to a target config -- and how each compares to persistence.
#
# Keras (MLP/LSTM/Transformer) inference runs via ONNX Runtime for speed.
# Uses the top4 variant (all four model classes are trained for it).
#
# Runs are parallelized; each (model,benchmark) writes its own CSV, then all are
# concatenated -> results/forecasting/cross_config/cross_config_10M_modelcmp.csv,
# and analyze_cross_config.py is run per model.
#
# Usage:
#   ./run_model_comparison.sh                 # all benches, 4 models, ONNX, -P 12
#   MODELS="mlp transformer" ./run_model_comparison.sh
#   PAR=8 BENCHES="dacapo_avrora spec_505.mcf_r" ./run_model_comparison.sh
#   ONNX=0 ./run_model_comparison.sh          # keras-native (no ONNX)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WF_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_ROOT="$(cd "$WF_DIR/../../.." && pwd)"
PYTHON="$REPO_ROOT/.venv/bin/python3"
HARNESS="$SCRIPT_DIR/predict_cross_config.py"

export PYTHONPATH="$WF_DIR"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 \
       TF_CPP_MIN_LOG_LEVEL=3

VARIANT="${VARIANT:-top4}"
MODELS="${MODELS:-dt mlp lstm transformer}"
PAR="${PAR:-12}"
ONNX_FLAG="--onnx"; [ "${ONNX:-1}" = "0" ] && ONNX_FLAG=""

# Counters translated S->C before forecasting. ref_cycles closes cross-frequency;
# adding cpu_cycles closes the cross-core gap (freq-invariant, core-dependent).
# The output CSV is tagged _refcpu when cpu_cycles is included so the ref-only and
# ref+cpu runs don't overwrite each other.
TRANSLATE="${TRANSLATE:-ref_cycles cpu_cycles}"
TAG=""; echo "$TRANSLATE" | grep -q cpu_cycles && TAG="_refcpu"

OUT="$REPO_ROOT/results/forecasting/cross_config/cross_config_10M_modelcmp${TAG}.csv"
mkdir -p "$(dirname "$OUT")"

# Discover benchmark 'rest' names from the cpu0 top4 forecasters (H1/T5 index).
REF_DIR="$REPO_ROOT/results/forecasting/models_10M/x86_desktop_heterogeneous_cpu0_${VARIANT}/4.0GHz/horizon_1/timesteps_5"
if [ -z "${BENCHES:-}" ]; then
    BENCHES=$(find "$REF_DIR" -maxdepth 1 \( -name '*_dt.pkl' -o -name '*_dt.joblib' \) 2>/dev/null \
        | sed -E 's#.*/aligned_(.+)_4\.0GHz_cpu0_dt\.(pkl|joblib)#\1#' | sort -u)
fi

TMPD="$(mktemp -d)"
trap 'rm -rf "$TMPD"' EXIT

echo "=== Cross-config model comparison (h=1, variant=$VARIANT) ==="
echo "  benches   : $(echo "$BENCHES" | wc -w)"
echo "  models    : $MODELS"
echo "  translate : $TRANSLATE"
echo "  onnx      : ${ONNX:-1}   parallelism: $PAR"
echo "  output    : $OUT"
echo

# Build the (model, benchmark) job list.
JOBS="$TMPD/jobs.txt"; : > "$JOBS"
for m in $MODELS; do for b in $BENCHES; do echo "$m $b" >> "$JOBS"; done; done

run_one() {
    local m="$1" b="$2"
    "$PYTHON" "$HARNESS" --benchmark "$b" --model "$m" --variant "$VARIANT" \
        --translate $TRANSLATE \
        --horizon 1 --timesteps 5 $ONNX_FLAG --out "$TMPD/${m}__${b}.csv" >/dev/null 2>&1
    echo "  done: $m | $b"
}
export -f run_one
export PYTHON HARNESS VARIANT ONNX_FLAG TMPD TRANSLATE

# Parallel dispatch (each job writes its own CSV -> no append races).
cat "$JOBS" | xargs -P "$PAR" -L 1 bash -c 'run_one "$0" "$1"'

# Concatenate all per-job CSVs into one.
first=1
: > "$OUT"
for f in "$TMPD"/*.csv; do
    [ -f "$f" ] || continue
    if [ "$first" = 1 ]; then cat "$f" > "$OUT"; first=0; else tail -n +2 "$f" >> "$OUT"; fi
done
echo
echo "Wrote $(($(wc -l < "$OUT") - 1)) rows -> $OUT"

# Per-model analysis (source->target matrices).
for m in $MODELS; do
    echo; echo "########## model = $m ##########"
    "$PYTHON" "$SCRIPT_DIR/analyze_cross_config.py" --csv "$OUT" --model "$m" || true
done

# Cross-model h=1 summary: which model best predicts the next timestep, and how
# each compares to persistence.
echo; echo "########## cross-model summary (h=1) ##########"
"$PYTHON" "$SCRIPT_DIR/analyze_cross_config.py" --csv "$OUT" --summary --horizon 1 || true

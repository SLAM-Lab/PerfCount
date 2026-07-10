#!/bin/bash
# run_inference.sh
# ================
# Single-sample inference latency benchmark for the workload-forecasting models
# (DT / MLP / LSTM / Transformer) across the three platforms (arm_server,
# x86_pcore, x86_ecore) and both feature sets (all vs top4).
#
# All models are timed on THIS host, single-threaded, pinned to one core. The
# `platform` label identifies the trained model set, not native hardware; arm and
# e-core numbers are a portability approximation from this host.
#
# Usage:
#   ./run_inference.sh                       # 10M, all platforms + feature sets
#   ./run_inference.sh 10M                   # explicit granularity
#   CPU=3 ./run_inference.sh                 # pin to a different core
#   PLATFORM=x86_pcore ./run_inference.sh    # restrict platform
#   FEATURE_SET=top4 ./run_inference.sh      # restrict feature set
#   SKIP_INSTALL=1 ./run_inference.sh        # skip the backend-dependency install
#
# On first run it installs the optional ONNX backend (ONNX Runtime + the skl2onnx /
# tf2onnx converters) into the repo .venv. The native backends (sklearn, keras) are
# assumed already present.
#
# Output:
#   results/forecasting/latency/inference_latency_{gran}.csv
#   results/forecasting/latency/pareto_{gran}.csv

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
PYTHON="$REPO_ROOT/.venv/bin/python3"
MEASURE="$SCRIPT_DIR/measure_inference_latency.py"

CPU="${CPU:-0}"
GRAN="${1:-10M}"

# Optional ONNX-backend packages (import name -> pip spec). Only the ones that
# fail to import are installed, so re-runs are cheap.
BACKEND_PKGS=("onnxruntime:onnxruntime" "skl2onnx:skl2onnx" "tf2onnx:tf2onnx")

if [ -z "${SKIP_INSTALL:-}" ]; then
    missing=()
    for entry in "${BACKEND_PKGS[@]}"; do
        mod="${entry%%:*}"; spec="${entry##*:}"
        if ! "$PYTHON" -c "import $mod" >/dev/null 2>&1; then
            missing+=("$spec")
        fi
    done
    if [ "${#missing[@]}" -gt 0 ]; then
        echo "=== Installing inference backends: ${missing[*]} ==="
        "$PYTHON" -m pip install --quiet "${missing[@]}" \
            || echo "[WARN] backend install failed; those backends will be skipped."
        echo
    fi
fi

# Single-threaded numeric backends for clean latency numbers.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1

EXTRA=()
[ -n "${PLATFORM:-}" ]    && EXTRA+=(--platform "$PLATFORM")
[ -n "${FEATURE_SET:-}" ] && EXTRA+=(--feature_set "$FEATURE_SET")

echo "=== WF inference latency benchmark ==="
echo "  Granularity : $GRAN"
echo "  CPU pin     : $CPU"
echo "  Platform    : ${PLATFORM:-all}"
echo "  Feature set : ${FEATURE_SET:-all}"
echo "  Output      : $REPO_ROOT/results/forecasting/latency"
echo

"$PYTHON" "$MEASURE" --gran "$GRAN" --cpu "$CPU" "${EXTRA[@]}"

echo
echo "Done."

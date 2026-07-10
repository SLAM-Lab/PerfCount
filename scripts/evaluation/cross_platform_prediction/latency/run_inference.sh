#!/bin/bash
# run_inference.sh
# ================
# Runs the three-backend inference latency benchmark for cross-platform
# CatBoost models.  All jobs run single-threaded, pinned to CPU 0.
#
# Platform scope (default): x86 desktop only.
#   - cross_freq / cross_proc: x86_{gran} (P-Core/E-Core), excludes x86_server
#     and every Arm platform.
#   - cross_sys: x86_{gran}, which bundles all four Xeon<->P-Core/E-Core
#     directions -- i.e. the x86 cross-system eval.
#   Set PLATFORM=all to discover every platform instead (see
#   benchmark_backends.py --platform).
#
# Usage:
#   ./run_inference.sh                # 10M, 100M, 1000M
#   ./run_inference.sh 10M            # single granularity
#   CPU=2 ./run_inference.sh          # pin to a different core
#   PLATFORM=all ./run_inference.sh   # discover every platform
#
# Output: results/cross_platform/inference/backends_{gran}.csv

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
PYTHON="$REPO_ROOT/.venv/bin/python3"
BENCH="$SCRIPT_DIR/benchmark_backends.py"

OUT_DIR="$REPO_ROOT/results/cross_platform/inference"
CACHE_DIR="/tmp/inference_so_cache"
CPU="${CPU:-0}"
PLATFORM="${PLATFORM:-x86}"

# Granularities to run (override with positional arg)
if [ $# -ge 1 ]; then
    GRANS=("$@")
else
    GRANS=(10M 100M 1000M)
fi

echo "=== Inference backend benchmark ==="
echo "  Output   : $OUT_DIR"
echo "  CPU pin  : $CPU"
echo "  Platform : $PLATFORM"
echo "  Grans    : ${GRANS[*]}"
echo

for gran in "${GRANS[@]}"; do
    echo "--- $gran ---"
    "$PYTHON" "$BENCH" \
        --gran "$gran" \
        --results_root "$REPO_ROOT/results/cross_platform" \
        --out_dir "$OUT_DIR" \
        --cache_dir "$CACHE_DIR" \
        --cpu "$CPU" \
        --platform "$PLATFORM"
    echo
done

echo "All done. Results in $OUT_DIR"

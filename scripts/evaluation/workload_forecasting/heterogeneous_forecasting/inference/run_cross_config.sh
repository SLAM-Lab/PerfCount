#!/bin/bash
# run_cross_config.sh
# ===================
# Cross-config workload-forecasting sweep: for every benchmark, run
# predict_cross_config.py (which evaluates all 56 source->target config pairs)
# across the requested models / horizons / timesteps, accumulating one CSV.
#
# translate-then-forecast vs oracle / naive / persistence, MAPE against the
# target config's aligned ground truth.
#
# Usage:
#   ./run_cross_config.sh                     # default: dt, H=1, T=5, all benches
#   MODELS="dt mlp" HORIZONS="1 5" ./run_cross_config.sh
#   BENCHES="dacapo_avrora spec_505.mcf_r" ./run_cross_config.sh
#
# Output: results/forecasting/cross_config/cross_config_10M.csv

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WF_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_ROOT="$(cd "$WF_DIR/../../.." && pwd)"
PYTHON="$REPO_ROOT/.venv/bin/python3"
HARNESS="$SCRIPT_DIR/predict_cross_config.py"

export PYTHONPATH="$WF_DIR"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1

MODELS="${MODELS:-dt}"
HORIZONS="${HORIZONS:-1}"
TIMESTEPS="${TIMESTEPS:-5}"

VARIANT="${VARIANT:-top4}"
TRANSLATE="${TRANSLATE:-ref_cycles}"
TAG=""; echo "$TRANSLATE" | grep -q cpu_cycles && TAG="_refcpu"
OUT="$REPO_ROOT/results/forecasting/cross_config/cross_config_10M_${VARIANT}${TAG}.csv"

# Discover benchmark 'rest' names from the cpu0 forecasters of this variant (H1/T5 index).
REF_DIR="$REPO_ROOT/results/forecasting/models_10M/x86_desktop_heterogeneous_cpu0_${VARIANT}/4.0GHz/horizon_1/timesteps_5"
if [ -z "${BENCHES:-}" ]; then
    BENCHES=$(find "$REF_DIR" -maxdepth 1 \( -name '*_dt.pkl' -o -name '*_dt.joblib' \) 2>/dev/null \
        | sed -E 's#.*/aligned_(.+)_4\.0GHz_cpu0_dt\.(pkl|joblib)#\1#' | sort -u || true)
fi

n_bench=$(echo "$BENCHES" | wc -w)
echo "=== Cross-config forecasting sweep ==="
echo "  benches   : $n_bench"
echo "  models    : $MODELS"
echo "  horizons  : $HORIZONS"
echo "  timesteps : $TIMESTEPS"
echo "  output    : $OUT"
echo

# Fresh accumulation.
rm -f "$OUT"

for m in $MODELS; do
  for h in $HORIZONS; do
    for t in $TIMESTEPS; do
      for b in $BENCHES; do
        echo "--- $b | $m | H:$h | T:$t ---"
        "$PYTHON" "$HARNESS" --benchmark "$b" --model "$m" --variant "$VARIANT" \
            --translate $TRANSLATE --horizon "$h" --timesteps "$t" --out "$OUT" 2>/dev/null \
            | grep -E "MAPE" || echo "  (no results)"
      done
    done
  done
done

echo
echo "Done. Rows in $OUT: $(($(wc -l < "$OUT") - 1))"

# Auto-analyze (first model) into the source->target matrices.
first_model=$(echo $MODELS | awk '{print $1}')
echo
"$PYTHON" "$SCRIPT_DIR/analyze_cross_config.py" --csv "$OUT" --model "$first_model" || true
echo
echo "Re-analyze any model with: ./analyze_cross_config.py --model <dt|mlp|lstm|transformer>"

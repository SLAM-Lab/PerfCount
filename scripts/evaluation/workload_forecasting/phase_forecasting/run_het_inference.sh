#!/bin/bash
# run_het_inference.sh  (C4 sweep)
# ================================
# Phase-aware HETEROGENEOUS INFERENCE across the multi-phase benches: train per-phase
# + delta forecasters on target config C (cpu0 @ 4.0GHz), predict C's future while
# observing a foreign config S -- raw (naive) or translated to C. Honest baseline is
# translated-persistence. Sweeps one or more SOURCE configs (cpu:freq).
#
# Usage:
#   ./run_het_inference.sh                       # default sources: cross-proc + cross-freq
#   SOURCES="16:4.0 0:1.0" MODEL=dt ./run_het_inference.sh
#   BENCHES="spec_505.mcf_r spec_525.x264_r" ./run_het_inference.sh
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WF_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"; REPO_ROOT="$(cd "$WF_DIR/../../.." && pwd)"
PYTHON="$REPO_ROOT/.venv/bin/python3"; HARNESS="$SCRIPT_DIR/predict_het_inference.py"
export PYTHONPATH="$WF_DIR"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
       TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 TF_CPP_MIN_LOG_LEVEL=3

FREQ="${FREQ:-4.0}"; CPU="${CPU:-0}"; MODEL="${MODEL:-dt}"; DELTA="${DELTA:-1}"
DELTA_FLAG=""; [ "$DELTA" = "1" ] && DELTA_FLAG="--delta"
EXTRA=""; [ "$MODEL" != "dt" ] && EXTRA="--no_gate --epochs ${NN_EPOCHS:-30}"; [ "$MODEL" = "lstm" ] && EXTRA="$EXTRA --stateless"
[ "${SAVE:-0}" = "1" ] && EXTRA="$EXTRA --save_models_dir $REPO_ROOT/results/forecasting/phase_forecasting/models"
COUNTERS="${COUNTERS:-ref_cycles cpu_cycles branches instructions}"   # top4b
# Default: E-core@4.0 (cross-proc) and P-core@1.0 (cross-freq).
SOURCES="${SOURCES:-16:4.0 0:1.0}"; PAR="${PAR:-80}"
BENCHES="${BENCHES:-\
spec_508.namd_r spec_505.mcf_r spec_525.x264_r spec_723.llvm_r spec_520.omnetpp_r spec_721.gcc_r \
dacapo_h2 dacapo_pmd spec_523.xalancbmk_r spec_777.zstd_r spec_557.xz_r spec_519.lbm_r \
dacapo_lusearch spec_500.perlbench_r dacapo_xalan}"

OUT="$REPO_ROOT/results/forecasting/phase_forecasting/het_infer_${MODEL}$([ "$DELTA" = 1 ] && echo _delta).csv"
mkdir -p "$(dirname "$OUT")"; rm -f "$OUT"; TMPD="$(mktemp -d)"; trap 'rm -rf "$TMPD"' EXIT
echo "=== Het inference (target cpu$CPU@${FREQ}GHz, top4b, $MODEL delta=$DELTA) ==="
echo "  sources: $SOURCES   benches: $(echo $BENCHES|wc -w)"; echo

run_one() {
    local b="$1" s="$2" sfx="${1}__${2/:/_}"
    "$PYTHON" "$HARNESS" --benchmark "aligned_${b}_${FREQ}GHz_cpu${CPU}" --source "$s" \
        --model "$MODEL" --counters $COUNTERS $DELTA_FLAG $EXTRA \
        --out "$TMPD/${sfx}.csv" >/dev/null 2>&1
    echo "  done: $b  S=$s"
}
export -f run_one; export PYTHON HARNESS MODEL COUNTERS DELTA_FLAG EXTRA FREQ CPU TMPD
: > "$TMPD/jobs"; for b in $BENCHES; do for s in $SOURCES; do echo "$b $s" >> "$TMPD/jobs"; done; done
cat "$TMPD/jobs" | xargs -P "$PAR" -L 1 bash -c 'run_one "$0" "$1"'

first=1; : > "$OUT"
for f in "$TMPD"/*.csv; do [ -f "$f" ] || continue
  if [ "$first" = 1 ]; then cat "$f" > "$OUT"; first=0; else tail -n +2 "$f" >> "$OUT"; fi; done
echo; echo "Wrote $(($(wc -l < "$OUT")-1)) rows -> $OUT"; echo
"$PYTHON" "$SCRIPT_DIR/analyze_het_inference.py" --csv "$OUT" || true

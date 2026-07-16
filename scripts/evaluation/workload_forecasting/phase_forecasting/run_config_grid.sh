#!/bin/bash
# run_config_grid.sh — full (source->target) config-pair grid for het-inference.
# 8 configs (cpu{0,16} x {1.0,2.0,3.0,4.0}) -> 56 ordered pairs. Same-core pairs = DVFS,
# cross-core pairs = Scheduling (incl. cross_proc_freq). One CSV: het_infer_grid_<model>_delta.csv.
# Usage: MODEL=dt PAR=80 ./run_config_grid.sh
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; WF_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$WF_DIR/../../.." && pwd)"; PY="$REPO_ROOT/.venv/bin/python3"
export PYTHONPATH="$WF_DIR"; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 TF_CPP_MIN_LOG_LEVEL=3
MODEL="${MODEL:-dt}"; PAR="${PAR:-80}"; COUNTERS="ref_cycles cpu_cycles branches instructions"
# predict_het_inference hardcodes stateless=True, so NO --stateless flag here (it errors).
EXTRA="--no_gate"; [ "$MODEL" != dt ] && EXTRA="--no_gate --epochs ${NN_EPOCHS:-30}"
CONFIGS="0:1.0 0:2.0 0:3.0 0:4.0 16:1.0 16:2.0 16:3.0 16:4.0"
DATA="$REPO_ROOT/processed_data_10M/x86_desktop_heterogeneous"
BENCHES="${BENCHES:-$(find "$DATA" -name 'aligned_*_4.0GHz_cpu0_phase*.csv' | sed -E 's#.*/aligned_(.+)_4\.0GHz_cpu0_phase[0-9]+\.csv#\1#' | sort -u)}"
OUT="$REPO_ROOT/results/forecasting/phase_forecasting/het_infer_grid_${MODEL}_delta.csv"
TMPD="$(mktemp -d)"; trap 'rm -rf "$TMPD"' EXIT
run_one(){ local tc=$1 tf=$2 s=$3 b=$4
  "$PY" "$SCRIPT_DIR/predict_het_inference.py" --benchmark "aligned_${b}_${tf}GHz_cpu${tc}" --source "$s" \
    --model "$MODEL" --counters $COUNTERS --delta $EXTRA \
    --out "$TMPD/${b}__T${tc}_${tf}__S${s/:/_}.csv" >/dev/null 2>&1; }
export -f run_one; export PY SCRIPT_DIR MODEL COUNTERS EXTRA TMPD
: > "$TMPD/jobs"
for T in $CONFIGS; do tc=${T%:*}; tf=${T#*:}
  for S in $CONFIGS; do [ "$S" = "$T" ] && continue
    for b in $BENCHES; do echo "$tc $tf $S $b" >> "$TMPD/jobs"; done; done; done
echo "grid jobs: $(wc -l < "$TMPD/jobs")  (model=$MODEL par=$PAR)"
cat "$TMPD/jobs" | xargs -P "$PAR" -L 1 bash -c 'run_one "$@"' _
first=1; : > "$OUT"
for f in "$TMPD"/*.csv; do [ -f "$f" ] || continue
  if [ "$first" = 1 ]; then cat "$f" > "$OUT"; first=0; else tail -n +2 "$f" >> "$OUT"; fi; done
echo "wrote $(($(wc -l < "$OUT")-1)) rows -> $OUT"

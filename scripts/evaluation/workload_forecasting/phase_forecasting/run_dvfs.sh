#!/bin/bash
# run_dvfs.sh -- DVFS (same-core cross-frequency) workload-forecasting sweep.
# =========================================================================
# For BOTH x86 cores (P-core cpu0, E-core cpu16), forecast every off-diagonal
# same-core (source_freq -> target_freq) pair: observe a foreign frequency S on
# the core, translate it to the target frequency C, forecast C's next-interval
# ref_cycles, and score vs translated-persistence. Per core there are 4x3 = 12
# ordered pairs (24 total). Reuses predict_het_inference.py unchanged; each run
# emits phase-UNAWARE (global) + phase-AWARE (per_phase / per_phase_gated) +
# translated-persistence, on the config-invariant top4b counter set.
#
# The homogeneous diagonal (source == target) is intentionally excluded here --
# that is plain same-config forecasting and comes from run_all_models.sh.
#
# Usage:
#   ./run_dvfs.sh                                   # all models, all benches, both cores
#   MODELS=dt ./run_dvfs.sh                         # DT-only fast pass (lands first)
#   BENCHES="spec_505.mcf_r spec_525.x264_r" MODELS=dt CORES="0 16" ./run_dvfs.sh   # smoke
# Env: MODELS, CORES, FREQS, BENCHES, PAR, NN_EPOCHS, SAVE, COUNTERS.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; WF_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$WF_DIR/../../.." && pwd)"; PY="$REPO_ROOT/.venv/bin/python3"
export PYTHONPATH="$WF_DIR"; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 TF_CPP_MIN_LOG_LEVEL=3

MODELS="${MODELS:-dt mlp lstm transformer}"; PAR="${PAR:-80}"
CORES="${CORES:-0 16}"; FREQS="${FREQS:-1.0 2.0 3.0 4.0}"
COUNTERS="${COUNTERS:-ref_cycles cpu_cycles branches instructions}"   # top4b
DATA="$REPO_ROOT/processed_data_10M/x86_desktop_heterogeneous"
OUTDIR="$REPO_ROOT/results/forecasting/phase_forecasting"; mkdir -p "$OUTDIR"

# Discover ALL benchmarks that exist at 4.0GHz on cpu0 (target-name stems), unless pinned.
BENCHES="${BENCHES:-$(find "$DATA" -name 'aligned_*_4.0GHz_cpu0_phase*.csv' \
  | sed -E 's#.*/aligned_(.+)_4\.0GHz_cpu0_phase[0-9]+\.csv#\1#' | sort -u)}"
NB=$(echo $BENCHES | wc -w)

# Build the same-core off-diagonal (target_cpu target_freq source_freq) triples.
PAIRS=""
for c in $CORES; do for tf in $FREQS; do for sf in $FREQS; do
  [ "$sf" = "$tf" ] && continue; PAIRS="$PAIRS ${c}:${tf}:${sf}"; done; done; done
NPAIRS=$(echo $PAIRS | wc -w)

echo "############################################################################"
echo "# run_dvfs : same-core cross-frequency forecasting (top4b, --delta)"
echo "#   cores: [$CORES]  freqs: [$FREQS]  pairs/core: $((NPAIRS/$(echo $CORES|wc -w)))  total pairs: $NPAIRS"
echo "#   benches: $NB   models: [$MODELS]   par: $PAR"
echo "############################################################################"; echo

run_one(){ local c=$1 tf=$2 sf=$3 b=$4
  "$PY" "$SCRIPT_DIR/predict_het_inference.py" \
    --benchmark "aligned_${b}_${tf}GHz_cpu${c}" --source "${c}:${sf}" \
    --model "$MODEL" --counters $COUNTERS --delta $EXTRA \
    --out "$TMPD/${b}__T${c}_${tf}__S${sf}.csv" >/dev/null 2>&1; }
export -f run_one; export PY SCRIPT_DIR COUNTERS

for MODEL in $MODELS; do
  EXTRA="--no_gate"; [ "$MODEL" != dt ] && EXTRA="--no_gate --epochs ${NN_EPOCHS:-30}"
  [ "${SAVE:-0}" = 1 ] && EXTRA="$EXTRA --save_models_dir $OUTDIR/models"
  export MODEL EXTRA
  TMPD="$(mktemp -d)"; export TMPD
  : > "$TMPD/jobs"
  for p in $PAIRS; do IFS=: read -r c tf sf <<< "$p"
    for b in $BENCHES; do echo "$c $tf $sf $b" >> "$TMPD/jobs"; done; done
  echo ">>> model=$MODEL : $(wc -l < "$TMPD/jobs") jobs (par=$PAR)"
  cat "$TMPD/jobs" | xargs -P "$PAR" -L 1 bash -c 'run_one "$@"' _
  OUT="$OUTDIR/dvfs_forecast_${MODEL}_delta.csv"
  first=1; : > "$OUT"
  for f in "$TMPD"/*.csv; do [ -f "$f" ] || continue
    if [ "$first" = 1 ]; then cat "$f" > "$OUT"; first=0; else tail -n +2 "$f" >> "$OUT"; fi; done
  echo "    wrote $(($(wc -l < "$OUT")-1)) rows -> $OUT"; echo
  rm -rf "$TMPD"
done
echo "############### run_dvfs complete -> $OUTDIR/dvfs_forecast_*_delta.csv ###############"

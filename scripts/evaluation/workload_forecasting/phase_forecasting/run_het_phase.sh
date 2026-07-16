#!/bin/bash
# run_het_phase.sh
# ================
# Phase-aware forecasting under a HETEROGENEOUS-HISTORY training regime: a fraction
# of each target config's TRAINING windows are replaced by donor windows collected at
# a different config, either NAIVE (raw swap) or TRANSLATED (donor unified to the
# target config via the CatBoost cross-freq / cross-proc models). Test is always on
# the target config's own future. Compares three regimes per benchmark:
#     homogeneous  (no injection; the floor)
#     naive        (raw donor injection)
#     translated   (donor injected after translation to the target config)
# each with the full phase-aware method set (global / per_phase / per_phase_gated /
# persistence) and the delta / transition-weighted machinery.
#
# Target config: cpu0 @ 4.0GHz. Donors: MODE=cross_freq (other freqs, same core) or
# cross_proc (other core). DT + --delta by default (the persistence-beating recipe).
#
# Usage:
#   ./run_het_phase.sh
#   MODE=cross_proc HET_PROB=0.3 ./run_het_phase.sh
#   BENCHES="spec_505.mcf_r spec_525.x264_r" MODEL=dt ./run_het_phase.sh

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WF_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$WF_DIR/../../.." && pwd)"
PYTHON="$REPO_ROOT/.venv/bin/python3"
HARNESS="$SCRIPT_DIR/predict_phase_forecast.py"
export PYTHONPATH="$WF_DIR"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 TF_CPP_MIN_LOG_LEVEL=3

DATASET="${DATASET:-x86_desktop_heterogeneous}"; CPU="${CPU:-0}"; FREQ="${FREQ:-4.0}"
MODEL="${MODEL:-dt}"; CLASSIFIER="${CLASSIFIER:-gmm}"; PHASE_COUNT="${PHASE_COUNT:-6}"; TIMESTEPS="${TIMESTEPS:-5}"
COUNTERS="${COUNTERS:-ref_cycles cpu_cycles branches instructions}"   # top4b (config-invariant)
DELTA="${DELTA:-1}"; DELTA_FLAG=""; [ "$DELTA" = "1" ] && DELTA_FLAG="--delta"
MODE="${MODE:-cross_freq}"; HET_PROB="${HET_PROB:-0.5}"; PAR="${PAR:-80}"
EXTRA=""; [ "$MODEL" != "dt" ] && EXTRA="--no_gate --epochs ${NN_EPOCHS:-30}"; [ "$MODEL" = "lstm" ] && EXTRA="$EXTRA --stateless"
# SAVE=1 -> persist trained ensembles under results/.../phase_forecasting/models/.
[ "${SAVE:-0}" = "1" ] && EXTRA="$EXTRA --save_models_dir $REPO_ROOT/results/forecasting/phase_forecasting/models"

# Translator roots (per target cpu for cross_freq; cross-proc tree + counter tree for cross_proc).
CF_ROOT="$REPO_ROOT/results/cross_platform/cross_freq/x86_10M/cpu${CPU}"
CP_ROOT="$REPO_ROOT/results/cross_platform/cross_proc/x86_10M"
CP_CTR="$CP_ROOT/counter_translation"
if [ "$MODE" = "cross_freq" ]; then TRANS_ARG="--cbm_model_dir $CF_ROOT"
else TRANS_ARG="--cbm_cross_proc_dir $CP_ROOT --cbm_cross_proc_counter_dir $CP_CTR"; fi

# 15 multi-phase benches by default (where phase-awareness matters).
BENCHES="${BENCHES:-\
spec_508.namd_r spec_505.mcf_r spec_525.x264_r spec_723.llvm_r spec_520.omnetpp_r spec_721.gcc_r \
dacapo_h2 dacapo_pmd spec_523.xalancbmk_r spec_777.zstd_r spec_557.xz_r spec_519.lbm_r \
dacapo_lusearch spec_500.perlbench_r dacapo_xalan}"

OUT="$REPO_ROOT/results/forecasting/phase_forecasting/het_phase_${MODE}_${MODEL}$([ "$DELTA" = 1 ] && echo _delta)_p${HET_PROB}.csv"
mkdir -p "$(dirname "$OUT")"; TMPD="$(mktemp -d)"; trap 'rm -rf "$TMPD"' EXIT

echo "=== Heterogeneous-history phase forecasting (cpu$CPU @ ${FREQ}GHz) ==="
echo "  mode: $MODE   prob: $HET_PROB   model: $MODEL   delta: $DELTA   benches: $(echo $BENCHES|wc -w)"
echo "  output: $OUT"; echo

run_one() {
    local b="$1" wl="aligned_${1}_${FREQ}GHz_cpu${CPU}"
    local c="$PYTHON $HARNESS --benchmark $wl --dataset $DATASET --input_counters $COUNTERS \
        --model $MODEL --classifier $CLASSIFIER --phase_count $PHASE_COUNT --timesteps $TIMESTEPS $DELTA_FLAG $EXTRA"
    # three regimes -> per-(bench,regime) temp CSV
    $c --out "$TMPD/${b}__homogeneous.csv" >/dev/null 2>&1
    $c --heterogeneous_prob $HET_PROB --heterogeneous_mode $MODE --out "$TMPD/${b}__naive.csv" >/dev/null 2>&1
    $c --heterogeneous_prob $HET_PROB --heterogeneous_mode $MODE $TRANS_ARG --out "$TMPD/${b}__translated.csv" >/dev/null 2>&1
    echo "  done: $b"
}
export -f run_one
export PYTHON HARNESS DATASET COUNTERS MODEL CLASSIFIER PHASE_COUNT TIMESTEPS DELTA_FLAG EXTRA FREQ CPU MODE HET_PROB TRANS_ARG TMPD

printf '%s\n' $BENCHES | xargs -P "$PAR" -L 1 bash -c 'run_one "$0"'

first=1; : > "$OUT"
for f in "$TMPD"/*.csv; do [ -f "$f" ] || continue
  if [ "$first" = 1 ]; then cat "$f" > "$OUT"; first=0; else tail -n +2 "$f" >> "$OUT"; fi; done
echo; echo "Wrote $(($(wc -l < "$OUT")-1)) rows -> $OUT"; echo
"$PYTHON" "$SCRIPT_DIR/analyze_het_phase.py" --csv "$OUT" || true

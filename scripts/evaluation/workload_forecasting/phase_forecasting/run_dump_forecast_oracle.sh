#!/bin/bash
# run_dump_forecast_oracle.sh -- build the Forecast x Oracle prediction set.
#
# For every (bench, config) this walk-forward forecasts that config's OWN true
# ref_cycles series, with no cross-platform translation (source == target, so the
# translation step is the identity). That isolates forecast error from translation
# error, which no other policy in the grid can do: every other forecast number folds
# the two together.
#
# Like the reactive oracle, the result is a BOUND, not a deployable policy -- it needs
# the true history of configs the workload never ran on. It answers "how good is the
# forecaster if the cross-platform model were perfect?"
#
# Output: $OUT_DIR/{config}/{bench}_phase{n}.csv with Time_pred_{config} + Time_true_{config}
# Feed it to the simulator with --forecast_oracle_dir.
#
#   ./run_dump_forecast_oracle.sh
#   BENCHES="dacapo_avrora spec_505.mcf_r" ./run_dump_forecast_oracle.sh
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WF_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$WF_DIR/../../.." && pwd)"
PY="$REPO_ROOT/.venv/bin/python3"
export PYTHONPATH="$WF_DIR"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 TF_CPP_MIN_LOG_LEVEL=3

MODEL="${MODEL:-dt}"; METHOD="${METHOD:-per_phase}"; PAR="${PAR:-24}"
# GATE: none|phase|persist|both -- see dump_dvfs_forecast.py --gate
GATE="${GATE:-none}"; GATE_WINDOW="${GATE_WINDOW:-200}"; GATE_MARGIN="${GATE_MARGIN:-0.05}"
CORES="${CORES:-0 16}"; FREQS="${FREQS:-1.0 2.0 3.0 4.0}"
MAX_TRAIN="${MAX_TRAIN:-20000}"; BLOCK="${BLOCK:-4096}"; WARMUP="${WARMUP:-2048}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/results/scheduling/forecast_oracle_10M}"
SIM_IN="$REPO_ROOT/results/scheduling/speedup_full_v2/granular_phase_traces"

BENCHES="${BENCHES:-$(ls "$SIM_IN" 2>/dev/null | sed -nE 's/^speedups_P_4\.0GHz_(.+)_phase[0-9]+\.csv$/\1/p' | sort -u)}"
NB=$(echo $BENCHES | wc -w)
mkdir -p "$OUT_DIR"
echo "=== forecast-oracle dump: $NB benches x cores[$CORES] x freqs[$FREQS]  model=$MODEL par=$PAR"
echo "    out: $OUT_DIR"

run_one(){ local b=$1 c=$2 sf=$3
  "$PY" "$SCRIPT_DIR/dump_dvfs_forecast.py" --bench "$b" --core "$c" --source_freq "$sf" \
    --out_dir "$OUT_DIR" --model "$MODEL" --method "$METHOD" --mode self_forecast \
    --gate "$GATE" --gate_window "$GATE_WINDOW" --gate_margin "$GATE_MARGIN" \
    --max_train "$MAX_TRAIN" --block "$BLOCK" --warmup "$WARMUP" >/dev/null 2>&1 \
    && echo "  ok  : $b cpu$c @ ${sf}GHz" || echo "  FAIL: $b cpu$c @ ${sf}GHz"; }
export -f run_one; export PY SCRIPT_DIR OUT_DIR MODEL METHOD MAX_TRAIN BLOCK WARMUP GATE GATE_WINDOW GATE_MARGIN

TMPD="$(mktemp -d)"; trap 'rm -rf "$TMPD"' EXIT; : > "$TMPD/jobs"
for b in $BENCHES; do for c in $CORES; do for sf in $FREQS; do echo "$b $c $sf" >> "$TMPD/jobs"; done; done; done
echo "    jobs: $(wc -l < "$TMPD/jobs")"
cat "$TMPD/jobs" | xargs -P "$PAR" -L 1 bash -c 'run_one "$@"' _

echo
echo "=== coverage check (expect one dir per config, each with every bench-phase)"
for d in "$OUT_DIR"/*/; do
  [ -d "$d" ] && echo "  $(basename "$d"): $(ls "$d" | wc -l) files"
done
echo "=== feed to the sim with: --forecast_oracle_dir $OUT_DIR"

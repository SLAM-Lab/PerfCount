#!/bin/bash
# run_dump_dvfs.sh -- generate the walk-forward DVFS forecast prediction set for the
# scheduling simulator. For every sim workload x core x source-frequency, dumps the
# per-phase speedup files (cross_freq_precompute layout) under $OUT_DIR, so the sim's
# Model_Greedy_Oracle_{P,E} policy (fed via --cross_freq_{p,e}_pred_dir) becomes the
# workload-forecasting policy.
#
# Usage:
#   ./run_dump_dvfs.sh                        # all sim benches, both cores, all src freqs
#   BENCHES="spec_505.mcf_r dacapo_h2" ./run_dump_dvfs.sh
# Env: BENCHES, CORES, FREQS, MODEL, METHOD, PAR, OUT_DIR, MAX_TRAIN, BLOCK, WARMUP.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; WF_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$WF_DIR/../../.." && pwd)"; PY="$REPO_ROOT/.venv/bin/python3"
export PYTHONPATH="$WF_DIR"; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 TF_CPP_MIN_LOG_LEVEL=3

MODEL="${MODEL:-dt}"; METHOD="${METHOD:-per_phase}"; MODE="${MODE:-forecast}"; PAR="${PAR:-24}"
# GATE: none|phase|persist|both -- see dump_dvfs_forecast.py --gate
GATE="${GATE:-none}"; GATE_WINDOW="${GATE_WINDOW:-200}"; GATE_MARGIN="${GATE_MARGIN:-0.05}"
CORES="${CORES:-0 16}"; FREQS="${FREQS:-1.0 2.0 3.0 4.0}"
MAX_TRAIN="${MAX_TRAIN:-20000}"; BLOCK="${BLOCK:-4096}"; WARMUP="${WARMUP:-2048}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/results/scheduling/forecast_predictions_10M}"
# Enumerate benches from the trace dir the SIMULATOR actually reads. This used to point at
# granular_phase_traces_10M, a stale set with 44 benches and NO spec26, so a default-arg dump
# silently covered a subset and spec26 had to be dumped separately -- which is what caused the
# dump/sim race that collapsed spec26's forecast policies onto the oracle. Keep this in sync
# with run_dump_forecast_oracle.sh and run_dvfs_study.sh.
SIM_IN="$REPO_ROOT/results/scheduling/speedup_full_v2/granular_phase_traces"

# Sim workloads = distinct benches with oracle P@4.0GHz speedup files.
BENCHES="${BENCHES:-$(ls "$SIM_IN" 2>/dev/null | sed -nE 's/^speedups_P_4\.0GHz_(.+)_phase[0-9]+\.csv$/\1/p' | sort -u)}"
NB=$(echo $BENCHES | wc -w)
mkdir -p "$OUT_DIR"
echo "=== run_dump_dvfs : $NB benches x cores[$CORES] x freqs[$FREQS]  model=$MODEL method=$METHOD par=$PAR"
echo "    out: $OUT_DIR"

# XPROC=1 -> cross-proc (migration): target = the OTHER core.
run_one(){ local b=$1 c=$2 sf=$3 tc_arg=""
  if [ "${XPROC:-0}" = 1 ]; then local t=$([ "$c" = 0 ] && echo 16 || echo 0); tc_arg="--target_core $t"; fi
  "$PY" "$SCRIPT_DIR/dump_dvfs_forecast.py" --bench "$b" --core "$c" $tc_arg --source_freq "$sf" \
    --out_dir "$OUT_DIR" --model "$MODEL" --method "$METHOD" --mode "$MODE" --horizon "${HORIZON:-1}" \
    --gate "$GATE" --gate_window "$GATE_WINDOW" --gate_margin "$GATE_MARGIN" \
    --max_train "$MAX_TRAIN" --block "$BLOCK" --warmup "$WARMUP" >/dev/null 2>&1 \
    && echo "  done: $b cpu$c @ ${sf}GHz" || echo "  FAIL: $b cpu$c @ ${sf}GHz"; }
export -f run_one; export PY SCRIPT_DIR OUT_DIR MODEL METHOD MODE MAX_TRAIN BLOCK WARMUP XPROC HORIZON GATE GATE_WINDOW GATE_MARGIN

TMPD="$(mktemp -d)"; trap 'rm -rf "$TMPD"' EXIT; : > "$TMPD/jobs"
for b in $BENCHES; do for c in $CORES; do for sf in $FREQS; do echo "$b $c $sf" >> "$TMPD/jobs"; done; done; done
echo "    jobs: $(wc -l < "$TMPD/jobs")"
cat "$TMPD/jobs" | xargs -P "$PAR" -L 1 bash -c 'run_one "$@"' _
echo "=== done -> $OUT_DIR (speedups_from_{P,E}_<src>GHz/)"
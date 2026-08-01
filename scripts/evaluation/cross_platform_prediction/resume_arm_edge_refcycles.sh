#!/bin/bash
# resume_arm_edge_refcycles.sh
# ============================
# Finish the tail of run_arm_edge_refcycles.sh after it was interrupted.
#
# Already complete (ref_cycles, verified by mtime 2026-07-20):
#   cross_freq cpu4          : spec_2017, spec_2026   (full, top4, both general)
#   cross_proc cpu4->cpu1    : spec_2017              (full, top4, both general)
#   cross_proc cpu1->cpu4    : spec_2017              (full, top4, both general)
#   cross_proc cpu4->cpu1    : spec_2026              (full, top4 only)
#
# Remaining, done here:
#   1. cross_proc cpu4->cpu1 spec_2026 : the two general models only
#   2. cross_proc cpu1->cpu4 spec_2026 : full + top4 + both general
#   3. dacapo_c1, all three axes       : full + top4 + both general
#
# dacapo_c1's existing `full` dirs are STALE cpu_cycles results (2026-07-06), so
# they are regenerated with --force rather than skipped.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PY="${PYTHON:-$REPO_ROOT/.venv/bin/python}"; [ -x "$PY" ] || PY="python3"

OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/results/cross_platform}"
CF="$SCRIPT_DIR/cross_frequency/cross_freq_arm.py"
CP="$SCRIPT_DIR/cross_processor/cross_proc_arm.py"
TOP="$SCRIPT_DIR/top_counters.py"

GRAN="${GRAN:-10M}"
DATA="$REPO_ROOT/processed_data_${GRAN}/arm_edge_heterogeneous"
JOBS="${JOBS:-8}"
TOPK_JOBS=$((JOBS * 3 / 2))

CFREQ="$OUT_ROOT/cross_freq/arm_edge_${GRAN}"
CPROC="$OUT_ROOT/cross_proc/arm_edge_${GRAN}"

# Both general models only; assumes a fresh ref_cycles `full` already exists.
general_only() {
    local full_dir="$1"; shift
    local counters
    counters=$("$PY" "$TOP" --results_dir "$full_dir" --top_k 4 2>/dev/null) || true
    if [ -z "$counters" ]; then echo "  [SKIP] no importance in $full_dir"; return; fi
    echo "  top4 (exactly 4 counters): $counters"
    local base="${full_dir%/full}"
    for gmode in general_insample general_temporal; do
        echo ""; echo "  [general] $gmode (top-4) -> $base/$gmode"
        # shellcheck disable=SC2086
        "$@" --out_dir "$base/$gmode" --input_counters $counters --mode "$gmode" --force
    done
}

# full (all-counter LOOCV, ref_cycles) -> exactly-top-4 LOOCV + both general models.
run_full_topk_general() {
    local full_dir="$1"; shift
    echo ""; echo "  [full] ref_cycles all-counter LOOCV -> $full_dir"
    "$@" --out_dir "$full_dir" --jobs "$JOBS" --force
    local counters
    counters=$("$PY" "$TOP" --results_dir "$full_dir" --top_k 4 2>/dev/null) || true
    if [ -z "$counters" ]; then echo "  [SKIP] no importance in $full_dir"; return; fi
    echo "  top4 (exactly 4 counters): $counters"
    local base="${full_dir%/full}"
    # shellcheck disable=SC2086
    "$@" --out_dir "$base/top4" --input_counters $counters --jobs "$TOPK_JOBS" --force
    for gmode in general_insample general_temporal; do
        echo ""; echo "  [general] $gmode (top-4) -> $base/$gmode"
        # shellcheck disable=SC2086
        "$@" --out_dir "$base/$gmode" --input_counters $counters --mode "$gmode" --force
    done
}

echo "############ RESUME 1/5: Cross-Proc OoO->InO | spec_2026 | general models only ############"
general_only "$CPROC/cpu4_to_cpu1/spec_2026/full" \
    "$PY" "$CP" --data_dir "$DATA" --src_cpu 4 --tgt_cpu 1 --suite spec_2026 --strict_loocv

echo ""; echo "############ RESUME 2/5: Cross-Proc InO->OoO | spec_2026 | full pipeline ############"
run_full_topk_general "$CPROC/cpu1_to_cpu4/spec_2026/full" \
    "$PY" "$CP" --data_dir "$DATA" --src_cpu 1 --tgt_cpu 4 --suite spec_2026 --strict_loocv

echo ""; echo "############ RESUME 3/5: Cross-Freq cpu4 (A77) | dacapo_c1 | full pipeline ############"
run_full_topk_general "$CFREQ/cpu4/dacapo_c1/full" \
    "$PY" "$CF" --data_dir "$DATA" --target_cpu 4 --suite dacapo_c1 --strict_loocv

echo ""; echo "############ RESUME 4/5: Cross-Proc OoO->InO | dacapo_c1 | full pipeline ############"
run_full_topk_general "$CPROC/cpu4_to_cpu1/dacapo_c1/full" \
    "$PY" "$CP" --data_dir "$DATA" --src_cpu 4 --tgt_cpu 1 --suite dacapo_c1 --strict_loocv

echo ""; echo "############ RESUME 5/5: Cross-Proc InO->OoO | dacapo_c1 | full pipeline ############"
run_full_topk_general "$CPROC/cpu1_to_cpu4/dacapo_c1/full" \
    "$PY" "$CP" --data_dir "$DATA" --src_cpu 1 --tgt_cpu 4 --suite dacapo_c1 --strict_loocv

echo ""; echo "RESUME COMPLETE — all arm_edge ref_cycles models built."

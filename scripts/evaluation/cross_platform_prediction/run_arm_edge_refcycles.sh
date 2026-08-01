#!/bin/bash
# run_arm_edge_refcycles.sh
# =========================
# Build the arm_edge_heterogeneous CROSS-PLATFORM models that predict
# REF_CYCLES (time-proportional target on arm: ref = cpu_cycles * 2/freq,
# i.e. constant-2GHz ticks), matching the x86 setup. Mirrors run_x86.sh's
# run_topk_and_general: per axis/direction/suite it runs
#   full  = all-counter LOOCV (ref_cycles)          -> feature importance + baseline
#   top4  = EXACTLY 4 counters (top_counters top_k=4) LOOCV
#   general_insample  = train on all workloads, eval each (learnability ceiling)
#   general_temporal  = train on first --temporal_frac of every workload, test tail
#
# Axes:  cross-frequency cpu4 (A77 big, 2.0<->1.0) and
#        cross-processor  L<->B (cpu1<->cpu4, both directions).
# Suites: spec_2017, spec_2026, dacapo_c1.   Granularity: 10M.
#
# Runs SEQUENTIALLY on purpose: each general_insample model trains on all
# workloads at once (~10 GB RAM), so parallel suites/axes would exhaust memory.
#
# Overwrites the standard arm_edge_10M result dirs (the ref_cycles target now
# supersedes the old cpu_cycles models, matching x86 which targets ref_cycles).
# Redirect with OUT_ROOT=/path to preserve the old results.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PY="${PYTHON:-$REPO_ROOT/.venv/bin/python}"; [ -x "$PY" ] || PY="python3"

OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/results/cross_platform}"
CF="$SCRIPT_DIR/cross_frequency/cross_freq_arm.py"
CP="$SCRIPT_DIR/cross_processor/cross_proc_arm.py"
TOP="$SCRIPT_DIR/top_counters.py"

GRAN="${GRAN:-10M}"
# Override to re-do a subset, e.g. SUITES="spec_2017 dacapo_c1" ./run_arm_edge_refcycles.sh
read -r -a SUITES <<< "${SUITES:-spec_2017 spec_2026 dacapo_c1}"
DATA="$REPO_ROOT/processed_data_${GRAN}/arm_edge_heterogeneous"
JOBS="${JOBS:-8}"
TOPK_JOBS=$((JOBS * 3 / 2))

# full (all-counter LOOCV, ref_cycles) -> exactly-top-4 LOOCV + both general models.
run_full_topk_general() {
    local full_dir="$1"; shift
    local base="${full_dir%/full}"

    # PURGE before running. --force only bypasses the top-level grand_summary
    # skip; it does NOT clear the per-fold caches (try_load_model /
    # load_fold_if_done). Re-running into a directory that still holds .cbm
    # models from an earlier target silently reuses them: folds whose stale
    # model is loaded get dropped or scored against the wrong target, which
    # produced 11-of-22 folds at ~100% wMAPE (copy-baseline level) with
    # ref_cycles/instructions importance pinned to zero. Purging is what makes
    # the re-run honest.
    for v in full top4 general_insample general_temporal; do
        rm -rf "$base/$v"
    done

    echo ""; echo "  [full] ref_cycles all-counter LOOCV -> $full_dir"
    "$@" --out_dir "$full_dir" --jobs "$JOBS" --force

    local counters
    counters=$("$PY" "$TOP" --results_dir "$full_dir" --top_k 4 2>/dev/null) || true
    if [ -z "$counters" ]; then
        echo "  [SKIP] no importance in $full_dir"; return
    fi
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

for suite in "${SUITES[@]}"; do
    if [ ! -d "$DATA/$suite" ]; then
        echo "[SKIP] no data at $DATA/$suite"; continue
    fi

    echo ""; echo "############ arm_edge ($GRAN) | Cross-Freq cpu4 (A77) | $suite ############"
    run_full_topk_general "$OUT_ROOT/cross_freq/arm_edge_${GRAN}/cpu4/$suite/full" \
        "$PY" "$CF" --data_dir "$DATA" --target_cpu 4 --suite "$suite" --strict_loocv

    echo ""; echo "############ arm_edge ($GRAN) | Cross-Proc OoO->InO (cpu4->cpu1) | $suite ############"
    run_full_topk_general "$OUT_ROOT/cross_proc/arm_edge_${GRAN}/cpu4_to_cpu1/$suite/full" \
        "$PY" "$CP" --data_dir "$DATA" --src_cpu 4 --tgt_cpu 1 --suite "$suite" --strict_loocv

    echo ""; echo "############ arm_edge ($GRAN) | Cross-Proc InO->OoO (cpu1->cpu4) | $suite ############"
    run_full_topk_general "$OUT_ROOT/cross_proc/arm_edge_${GRAN}/cpu1_to_cpu4/$suite/full" \
        "$PY" "$CP" --data_dir "$DATA" --src_cpu 1 --tgt_cpu 4 --suite "$suite" --strict_loocv
done

echo ""; echo "All arm_edge ref_cycles models complete. Results under:"
echo "  $OUT_ROOT/cross_freq/arm_edge_${GRAN}/cpu4/<suite>/{full,top4,general_insample,general_temporal}"
echo "  $OUT_ROOT/cross_proc/arm_edge_${GRAN}/cpu{4_to_1,1_to_4}/<suite>/{full,top4,general_insample,general_temporal}"

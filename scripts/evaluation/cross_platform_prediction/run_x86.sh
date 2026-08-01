#!/bin/bash
# run_x86.sh
# ==========
# x86 cross-platform experiments: cross-frequency, cross-processor,
# and cross-system LOOCV with top-K ablations.
# All at 10M, 100M, and 1000M granularity.
#
# A suite is skipped automatically (with a [SKIP] message) if the
# corresponding processed_data/<suite> folder doesn't exist for the data
# dir(s) involved, so this script adapts as new data is collected without
# needing hardcoded per-platform suite lists.
#
# Usage:
#   ./run_x86.sh [section]   (default: all)
#
# Sections:
#   x86_cf        x86 desktop + server cross-frequency
#   x86_cp        x86 desktop cross-processor
#   x86_cs        x86 cross-system (server <-> desktop)
#   all           everything
#
# Override output root:
#   OUT_ROOT=/path  ./run_x86.sh
#
# Per-section suite/ablation rules:
#   Cross-frequency (SUITES_CF): dacapo_c1 and dacapo_c2 get a full run only
#     (no top-4/top-6 -- see cf_runner). spec_2017 and spec_2026 get full +
#     top-4 + top-6. x86
#     machines additionally get dacapo_c1_pruned and dacapo_c2_pruned, full
#     config only, dropping cassandra/tradebeans/tradesoap/h2o/kafka (see
#     shared_features.UNSTABLE_DACAPO_WORKLOADS) -- needed for the x86
#     DaCapo c1/c2 x full/pruned comparison figure.
#   Cross-processor (SUITES_CP): dacapo_c1, spec_2017, spec_2026 all get
#     full + top-4 + top-6. No dacapo_c2, no pruned variants.
#   Cross-system (SUITES_CS): dacapo_c1, spec_2017, spec_2026 get top-4 LOOCV
#     plus both general models (general_insample / general_temporal), all on the
#     same top-4 counters, selected from the existing full/ importance.
#     No dacapo_c2, no pruned variants.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
TARGET="${1:-all}"

# ---------------------------------------------------------------------------
# Python interpreter. Override with PYTHON=/path/to/venv/bin/python if the
# dependencies are not on the default python3 (e.g. you did not activate the
# virtualenv). Checked up front, because a missing pandas would otherwise
# surface much later as a misleading "No importance data" skip.
# ---------------------------------------------------------------------------
PY="${PYTHON:-python3}"
if ! "$PY" -c 'import pandas, catboost' 2>/dev/null; then
    echo "error: '$PY' cannot import pandas and catboost." >&2
    echo "       Activate the virtualenv, or run:  PYTHON=/path/to/python $0 $*" >&2
    echo "       Install with:  pip install -r requirements.txt" >&2
    exit 1
fi


OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/results/cross_platform}"

# Prediction target for the x86 machines. ref_cycles is a real hardware
# counter here (constant-rate, tracks wall-clock time), which is what the
# released x86 results use. Override with TARGET_KEY=cpu_cycles to predict
# core cycles instead -- comparable with the Arm desktop/server results but
# not with the released x86 numbers.
TARGET_KEY="${TARGET_KEY:-ref_cycles}"
CF_X86="$SCRIPT_DIR/cross_frequency/cross_freq_x86.py"
CP_X86="$SCRIPT_DIR/cross_processor/cross_proc_x86.py"
CS="$SCRIPT_DIR/cross_system/cross_sys_arm.py"
TOP_COUNTERS="$SCRIPT_DIR/top_counters.py"

TOP_K_VALUES=(4 6)
JOBS=8                          # trainer default parallel workers (full runs)
TOPK_JOBS=$(( JOBS * 3 / 2 ))   # top-K uses less memory/job -> 50% more parallelism
SUITES_CF=(dacapo_c1 spec_2017 spec_2026)  # dacapo_c2 commented out for now
SUITES_CP=(dacapo_c1 spec_2017 spec_2026)
SUITES_CS=(dacapo_c1 spec_2017 spec_2026)
PRUNED_SUITES=()  # dacapo_c1_pruned / dacapo_c2_pruned commented out for now (was: dacapo_c1 dacapo_c2)

# ---------------------------------------------------------------------------
# Helper: does processed_data/<suite> exist under a given data dir?
# ---------------------------------------------------------------------------
suite_dir_exists() {
    local data_dir="$1"
    local suite="$2"
    [ -d "$data_dir/$suite" ]
}

# ---------------------------------------------------------------------------
# Helper: run top-K ablation using importance from the full run.
# ---------------------------------------------------------------------------
run_topk() {
    local full_dir="$1"
    local top_k="$2"
    shift 2

    if [ ! -d "$full_dir" ]; then
        echo "  [SKIP] Full results dir not found: $full_dir"
        return
    fi

    local counters
    counters=$("$PY" "$TOP_COUNTERS" --results_dir "$full_dir" --top_k "$top_k" 2>/dev/null) || true
    if [ -z "$counters" ]; then
        echo "  [SKIP] No importance data in $full_dir"
        return
    fi

    local topk_dir="${full_dir%/full}/top${top_k}"
    echo "  top${top_k} counters: $counters"
    # shellcheck disable=SC2086
    "$@" --out_dir "$topk_dir" --input_counters $counters --jobs "$TOPK_JOBS"
}

# Helper: run the full config only -- no top-K ablation.
run_full_only() {
    local full_dir="$1"
    shift
    "$@" --out_dir "$full_dir"
}

# Helper: top-4 only (cross-freq + cross-proc). The full and top-6 runs are
# commented out per request -- we only run top-4. top-4 still reads its feature
# importance from the EXISTING full_dir results (run a prior full pass to
# populate them). Re-enable the lines below to restore full + top-4 + top-6.
run_with_ablation() {
    local full_dir="$1"
    shift
    # "$@" --out_dir "$full_dir"                 # full run (commented out)
    # for k in "${TOP_K_VALUES[@]}"; do          # top-4 + top-6 (commented out)
    #     run_topk "$full_dir" "$k" "$@"
    # done
    run_topk "$full_dir" 4 "$@"                   # top-4 only
}

# Helper: top-4 LOOCV PLUS the two general models -- ALL on the SAME top-4
# counters, so LOOCV vs general differ only in the train/test data split (not
# features). The general models train one model on all workloads (complement /
# ceiling to LOOCV) and land in sibling dirs next to full/ and top4/:
#   top4              = LOOCV, each workload held out
#   general_insample  = train on all, evaluate on each (learnability ceiling)
#   general_temporal  = train on first --temporal_frac of every workload, test tail
run_topk_and_general() {
    local full_dir="$1"
    shift

    # Full-counter LOOCV pass. This has to run first: the top-4 set below is
    # selected from its feature importances, so on a fresh OUT_ROOT every config
    # would otherwise skip with "No importance data". Already-complete runs are
    # detected by grand_summary.csv and skipped, so re-runs resume.
    if [ -f "$full_dir/grand_summary.csv" ]; then
        echo "  [SKIP] full pass already complete: $full_dir"
    else
        echo "  [full] -> $full_dir"
        "$@" --out_dir "$full_dir" --jobs "$JOBS"
    fi

    # top-4 counter set, selected once from the full importance and shared by
    # all three runs below.
    local counters
    counters=$("$PY" "$TOP_COUNTERS" --results_dir "$full_dir" --top_k 4 2>/dev/null) || true
    if [ -z "$counters" ]; then
        echo "  [SKIP] No importance data in $full_dir (full pass produced none)"
        return
    fi
    echo "  top4 counters: $counters"

    local base="${full_dir%/full}"
    # shellcheck disable=SC2086
    "$@" --out_dir "$base/top4" --input_counters $counters --jobs "$TOPK_JOBS"   # LOOCV top-4
    for gmode in general_insample general_temporal; do
        echo ""; echo "  [general] $gmode (top-4) -> $base/$gmode"
        # shellcheck disable=SC2086
        "$@" --out_dir "$base/$gmode" --input_counters $counters --mode "$gmode"
    done
}

# Helper: pick which runner a cross-frequency suite gets. All active suites
# (dacapo_c1, spec_2017, spec_2026) go through top-4 ablation + both general
# models -- dacapo_c1's top-4 reads importance from its existing full results.
cf_runner() {
    echo run_topk_and_general
}


# ###########################################################################
#                        X86 CROSS-FREQUENCY
# ###########################################################################
if [[ "$TARGET" == "x86_cf" || "$TARGET" == "all" ]]; then

    # -----------------------------------------------------------------------
    # X86 DESKTOP — P-Core (cpu0) and E-Core (cpu16)
    # -----------------------------------------------------------------------
    for gran in 10M; do  # 100M 1000M commented out for now -- 10M only
        data_dir="$REPO_ROOT/processed_data_${gran}/x86_desktop_heterogeneous"

        for suite in "${SUITES_CF[@]}"; do
            if ! suite_dir_exists "$data_dir" "$suite"; then
                echo "  [SKIP] x86_desktop ($gran) | Cross-Freq | $suite: no data at $data_dir/$suite"
                continue
            fi

            echo ""; echo "=== x86_desktop ($gran) | Cross-Freq | P-Core | $suite ==="
            "$(cf_runner "$suite")" "$OUT_ROOT/cross_freq/x86_${gran}/cpu0/$suite/full" \
                "$PY" "$CF_X86" --data_dir "$data_dir" --target_cpu 0 --suite "$suite" --strict_loocv --target_key "$TARGET_KEY"

            echo ""; echo "=== x86_desktop ($gran) | Cross-Freq | E-Core | $suite ==="
            "$(cf_runner "$suite")" "$OUT_ROOT/cross_freq/x86_${gran}/cpu16/$suite/full" \
                "$PY" "$CF_X86" --data_dir "$data_dir" --target_cpu 16 --suite "$suite" --strict_loocv --target_key "$TARGET_KEY"
        done

        # -------------------------------------------------------------------
        # DaCapo pruned variants (full only, no top-K ablation) -- dropping
        # cassandra/tradebeans/tradesoap/h2o/kafka. Needed for the x86
        # DaCapo c1/c2 x full/pruned comparison figure.
        # -------------------------------------------------------------------
        for suite in "${PRUNED_SUITES[@]}"; do
            if ! suite_dir_exists "$data_dir" "$suite"; then
                echo "  [SKIP] x86_desktop ($gran) | Cross-Freq | ${suite}_pruned: no data at $data_dir/$suite"
                continue
            fi

            echo ""; echo "=== x86_desktop ($gran) | Cross-Freq | P-Core | ${suite}_pruned ==="
            run_full_only "$OUT_ROOT/cross_freq/x86_${gran}/cpu0/${suite}_pruned/full" \
                "$PY" "$CF_X86" --data_dir "$data_dir" --target_cpu 0 --suite "$suite" --strict_loocv --exclude_unstable_dacapo --target_key "$TARGET_KEY"

            echo ""; echo "=== x86_desktop ($gran) | Cross-Freq | E-Core | ${suite}_pruned ==="
            run_full_only "$OUT_ROOT/cross_freq/x86_${gran}/cpu16/${suite}_pruned/full" \
                "$PY" "$CF_X86" --data_dir "$data_dir" --target_cpu 16 --suite "$suite" --strict_loocv --exclude_unstable_dacapo --target_key "$TARGET_KEY"
        done
    done

    # -----------------------------------------------------------------------
    # X86 SERVER  -- disabled for now (change `if false` to `if true` to re-enable)
    # -----------------------------------------------------------------------
    if false; then
    for gran in 10M; do  # 100M 1000M commented out for now -- 10M only
        data_dir="$REPO_ROOT/processed_data_${gran}/x86_server"

        for suite in "${SUITES_CF[@]}"; do
            if ! suite_dir_exists "$data_dir" "$suite"; then
                echo "  [SKIP] x86_server ($gran) | Cross-Freq | $suite: no data at $data_dir/$suite"
                continue
            fi
            echo ""; echo "=== x86_server ($gran) | Cross-Freq | $suite ==="
            "$(cf_runner "$suite")" "$OUT_ROOT/cross_freq/x86_server_${gran}/$suite/full" \
                "$PY" "$CF_X86" --data_dir "$data_dir" --suite "$suite" --strict_loocv --target_key "$TARGET_KEY"
        done

        # -------------------------------------------------------------------
        # DaCapo pruned variants (full only, no top-K ablation)
        # -------------------------------------------------------------------
        for suite in "${PRUNED_SUITES[@]}"; do
            if ! suite_dir_exists "$data_dir" "$suite"; then
                echo "  [SKIP] x86_server ($gran) | Cross-Freq | ${suite}_pruned: no data at $data_dir/$suite"
                continue
            fi
            echo ""; echo "=== x86_server ($gran) | Cross-Freq | ${suite}_pruned ==="
            run_full_only "$OUT_ROOT/cross_freq/x86_server_${gran}/${suite}_pruned/full" \
                "$PY" "$CF_X86" --data_dir "$data_dir" --suite "$suite" --strict_loocv --exclude_unstable_dacapo --target_key "$TARGET_KEY"
        done
    done
    fi  # end X86 SERVER (disabled)

fi


# ###########################################################################
#                        X86 CROSS-PROCESSOR
# ###########################################################################
if [[ "$TARGET" == "x86_cp" || "$TARGET" == "all" ]]; then

    for gran in 10M; do  # 100M 1000M commented out for now -- 10M only
        data_dir="$REPO_ROOT/processed_data_${gran}/x86_desktop_heterogeneous"

        for suite in "${SUITES_CP[@]}"; do
            if ! suite_dir_exists "$data_dir" "$suite"; then
                echo "  [SKIP] x86_desktop ($gran) | Cross-Proc | $suite: no data at $data_dir/$suite"
                continue
            fi

            echo ""; echo "=== x86_desktop ($gran) | Cross-Proc | P → E | $suite ==="
            run_topk_and_general "$OUT_ROOT/cross_proc/x86_${gran}/cpu0_to_cpu16/$suite/full" \
                "$PY" "$CP_X86" --data_dir "$data_dir" --src_cpu 0 --tgt_cpu 16 --suite "$suite" --strict_loocv --target_key "$TARGET_KEY"

            echo ""; echo "=== x86_desktop ($gran) | Cross-Proc | E → P | $suite ==="
            run_topk_and_general "$OUT_ROOT/cross_proc/x86_${gran}/cpu16_to_cpu0/$suite/full" \
                "$PY" "$CP_X86" --data_dir "$data_dir" --src_cpu 16 --tgt_cpu 0 --suite "$suite" --strict_loocv --target_key "$TARGET_KEY"
        done
    done

fi


# ###########################################################################
#                        X86 CROSS-SYSTEM
# ###########################################################################
# Restricted to matched 1.0GHz -> 1.0GHz for now (--freq 1.0) — cross-system
# already gives a 3x3 platform matrix per suite; sweeping every source x
# target frequency combination on top of that is not needed yet. Drop
# --freq 1.0 below to re-enable the full frequency sweep.
if [[ "$TARGET" == "x86_cs" || "$TARGET" == "all" ]]; then

    for gran in 10M; do  # 100M 1000M commented out for now -- 10M only
        SERVER="$REPO_ROOT/processed_data_${gran}/x86_server"
        DESKTOP="$REPO_ROOT/processed_data_${gran}/x86_desktop_heterogeneous"
        OUT="$OUT_ROOT/cross_sys/x86_${gran}"

        for suite in "${SUITES_CS[@]}"; do
            if ! suite_dir_exists "$SERVER" "$suite" || ! suite_dir_exists "$DESKTOP" "$suite"; then
                echo "  [SKIP] x86 ($gran) | Cross-Sys | server<->desktop | $suite: missing in $SERVER or $DESKTOP"
                continue
            fi

            echo ""; echo "=== x86 ($gran) | Cross-Sys | Server → P-Core | $suite ==="
            run_topk_and_general "$OUT/server_to_pcore/$suite/full" \
                "$PY" "$CS" --src_data_dir "$SERVER" --tgt_data_dir "$DESKTOP" \
                    --tgt_cpu 0 --src_label server --tgt_label pcore \
                    --suite "$suite" --target_key "$TARGET_KEY" --freq 1.0

            echo ""; echo "=== x86 ($gran) | Cross-Sys | Server → E-Core | $suite ==="
            run_topk_and_general "$OUT/server_to_ecore/$suite/full" \
                "$PY" "$CS" --src_data_dir "$SERVER" --tgt_data_dir "$DESKTOP" \
                    --tgt_cpu 16 --src_label server --tgt_label ecore \
                    --suite "$suite" --target_key "$TARGET_KEY" --freq 1.0

            echo ""; echo "=== x86 ($gran) | Cross-Sys | P-Core → Server | $suite ==="
            run_topk_and_general "$OUT/pcore_to_server/$suite/full" \
                "$PY" "$CS" --src_data_dir "$DESKTOP" --tgt_data_dir "$SERVER" \
                    --src_cpu 0 --src_label pcore --tgt_label server \
                    --suite "$suite" --target_key "$TARGET_KEY" --freq 1.0

            echo ""; echo "=== x86 ($gran) | Cross-Sys | E-Core → Server | $suite ==="
            run_topk_and_general "$OUT/ecore_to_server/$suite/full" \
                "$PY" "$CS" --src_data_dir "$DESKTOP" --tgt_data_dir "$SERVER" \
                    --src_cpu 16 --src_label ecore --tgt_label server \
                    --suite "$suite" --target_key "$TARGET_KEY" --freq 1.0
        done
    done

fi


echo ""; echo "All x86 models complete. Results in: $OUT_ROOT"

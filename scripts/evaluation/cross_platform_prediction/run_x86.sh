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
#   Cross-system (SUITES_CS): dacapo_c1, spec_2017, spec_2026 get a full
#     run only (no top-4/top-6). No dacapo_c2, no pruned variants.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
TARGET="${1:-all}"

OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/results/cross_platform}"
CF_X86="$SCRIPT_DIR/cross_frequency/cross_freq_x86.py"
CP_X86="$SCRIPT_DIR/cross_processor/cross_proc_x86.py"
CS="$SCRIPT_DIR/cross_system/cross_sys_arm.py"
TOP_COUNTERS="$SCRIPT_DIR/top_counters.py"

TOP_K_VALUES=(4 6)
SUITES_CF=(dacapo_c1 dacapo_c2 spec_2017 spec_2026)
SUITES_CP=(dacapo_c1 spec_2017 spec_2026)
SUITES_CS=(dacapo_c1 spec_2017 spec_2026)
PRUNED_SUITES=(dacapo_c1 dacapo_c2)

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
    counters=$(python3 "$TOP_COUNTERS" --results_dir "$full_dir" --top_k "$top_k" 2>/dev/null) || true
    if [ -z "$counters" ]; then
        echo "  [SKIP] No importance data in $full_dir"
        return
    fi

    local topk_dir="${full_dir%/full}/top${top_k}"
    echo "  top${top_k} counters: $counters"
    # shellcheck disable=SC2086
    "$@" --out_dir "$topk_dir" --input_counters $counters
}

# Helper: run the full config only -- no top-K ablation.
run_full_only() {
    local full_dir="$1"
    shift
    "$@" --out_dir "$full_dir"
}

# Helper: run full + top-4 + top-6 ablation.
run_with_ablation() {
    local full_dir="$1"
    shift
    "$@" --out_dir "$full_dir"
    for k in "${TOP_K_VALUES[@]}"; do
        run_topk "$full_dir" "$k" "$@"
    done
}

# Helper: pick which runner a cross-frequency suite gets. dacapo_c1/dacapo_c2
# are full-only there; everything else (spec_2017, spec_2026) gets ablation.
cf_runner() {
    case "$1" in
        dacapo_c1|dacapo_c2) echo run_full_only ;;
        *)                   echo run_with_ablation ;;
    esac
}


# ###########################################################################
#                        X86 CROSS-FREQUENCY
# ###########################################################################
if [[ "$TARGET" == "x86_cf" || "$TARGET" == "all" ]]; then

    # -----------------------------------------------------------------------
    # X86 DESKTOP — P-Core (cpu0) and E-Core (cpu16)
    # -----------------------------------------------------------------------
    for gran in 10M 100M 1000M; do
        data_dir="$REPO_ROOT/processed_data_${gran}/x86_desktop_heterogeneous"

        for suite in "${SUITES_CF[@]}"; do
            if ! suite_dir_exists "$data_dir" "$suite"; then
                echo "  [SKIP] x86_desktop ($gran) | Cross-Freq | $suite: no data at $data_dir/$suite"
                continue
            fi

            echo ""; echo "=== x86_desktop ($gran) | Cross-Freq | P-Core | $suite ==="
            "$(cf_runner "$suite")" "$OUT_ROOT/cross_freq/x86_${gran}/cpu0/$suite/full" \
                python3 "$CF_X86" --data_dir "$data_dir" --target_cpu 0 --suite "$suite" --strict_loocv

            echo ""; echo "=== x86_desktop ($gran) | Cross-Freq | E-Core | $suite ==="
            "$(cf_runner "$suite")" "$OUT_ROOT/cross_freq/x86_${gran}/cpu16/$suite/full" \
                python3 "$CF_X86" --data_dir "$data_dir" --target_cpu 16 --suite "$suite" --strict_loocv
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
                python3 "$CF_X86" --data_dir "$data_dir" --target_cpu 0 --suite "$suite" --strict_loocv --exclude_unstable_dacapo

            echo ""; echo "=== x86_desktop ($gran) | Cross-Freq | E-Core | ${suite}_pruned ==="
            run_full_only "$OUT_ROOT/cross_freq/x86_${gran}/cpu16/${suite}_pruned/full" \
                python3 "$CF_X86" --data_dir "$data_dir" --target_cpu 16 --suite "$suite" --strict_loocv --exclude_unstable_dacapo
        done
    done

    # -----------------------------------------------------------------------
    # X86 SERVER
    # -----------------------------------------------------------------------
    for gran in 10M 100M 1000M; do
        data_dir="$REPO_ROOT/processed_data_${gran}/x86_server"

        for suite in "${SUITES_CF[@]}"; do
            if ! suite_dir_exists "$data_dir" "$suite"; then
                echo "  [SKIP] x86_server ($gran) | Cross-Freq | $suite: no data at $data_dir/$suite"
                continue
            fi
            echo ""; echo "=== x86_server ($gran) | Cross-Freq | $suite ==="
            "$(cf_runner "$suite")" "$OUT_ROOT/cross_freq/x86_server_${gran}/$suite/full" \
                python3 "$CF_X86" --data_dir "$data_dir" --suite "$suite" --strict_loocv
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
                python3 "$CF_X86" --data_dir "$data_dir" --suite "$suite" --strict_loocv --exclude_unstable_dacapo
        done
    done

fi


# ###########################################################################
#                        X86 CROSS-PROCESSOR
# ###########################################################################
if [[ "$TARGET" == "x86_cp" || "$TARGET" == "all" ]]; then

    for gran in 10M 100M 1000M; do
        data_dir="$REPO_ROOT/processed_data_${gran}/x86_desktop_heterogeneous"

        for suite in "${SUITES_CP[@]}"; do
            if ! suite_dir_exists "$data_dir" "$suite"; then
                echo "  [SKIP] x86_desktop ($gran) | Cross-Proc | $suite: no data at $data_dir/$suite"
                continue
            fi

            echo ""; echo "=== x86_desktop ($gran) | Cross-Proc | P → E | $suite ==="
            run_with_ablation "$OUT_ROOT/cross_proc/x86_${gran}/cpu0_to_cpu16/$suite/full" \
                python3 "$CP_X86" --data_dir "$data_dir" --src_cpu 0 --tgt_cpu 16 --suite "$suite" --strict_loocv

            echo ""; echo "=== x86_desktop ($gran) | Cross-Proc | E → P | $suite ==="
            run_with_ablation "$OUT_ROOT/cross_proc/x86_${gran}/cpu16_to_cpu0/$suite/full" \
                python3 "$CP_X86" --data_dir "$data_dir" --src_cpu 16 --tgt_cpu 0 --suite "$suite" --strict_loocv
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

    for gran in 10M 100M 1000M; do
        SERVER="$REPO_ROOT/processed_data_${gran}/x86_server"
        DESKTOP="$REPO_ROOT/processed_data_${gran}/x86_desktop_heterogeneous"
        OUT="$OUT_ROOT/cross_sys/x86_${gran}"

        for suite in "${SUITES_CS[@]}"; do
            if ! suite_dir_exists "$SERVER" "$suite" || ! suite_dir_exists "$DESKTOP" "$suite"; then
                echo "  [SKIP] x86 ($gran) | Cross-Sys | server<->desktop | $suite: missing in $SERVER or $DESKTOP"
                continue
            fi

            echo ""; echo "=== x86 ($gran) | Cross-Sys | Server → P-Core | $suite ==="
            run_full_only "$OUT/server_to_pcore/$suite/full" \
                python3 "$CS" --src_data_dir "$SERVER" --tgt_data_dir "$DESKTOP" \
                    --tgt_cpu 0 --src_label server --tgt_label pcore \
                    --suite "$suite" --target_key ref_cycles --freq 1.0

            echo ""; echo "=== x86 ($gran) | Cross-Sys | Server → E-Core | $suite ==="
            run_full_only "$OUT/server_to_ecore/$suite/full" \
                python3 "$CS" --src_data_dir "$SERVER" --tgt_data_dir "$DESKTOP" \
                    --tgt_cpu 16 --src_label server --tgt_label ecore \
                    --suite "$suite" --target_key ref_cycles --freq 1.0

            echo ""; echo "=== x86 ($gran) | Cross-Sys | P-Core → Server | $suite ==="
            run_full_only "$OUT/pcore_to_server/$suite/full" \
                python3 "$CS" --src_data_dir "$DESKTOP" --tgt_data_dir "$SERVER" \
                    --src_cpu 0 --src_label pcore --tgt_label server \
                    --suite "$suite" --target_key ref_cycles --freq 1.0

            echo ""; echo "=== x86 ($gran) | Cross-Sys | E-Core → Server | $suite ==="
            run_full_only "$OUT/ecore_to_server/$suite/full" \
                python3 "$CS" --src_data_dir "$DESKTOP" --tgt_data_dir "$SERVER" \
                    --src_cpu 16 --src_label ecore --tgt_label server \
                    --suite "$suite" --target_key ref_cycles --freq 1.0
        done
    done

fi


echo ""; echo "All x86 models complete. Results in: $OUT_ROOT"

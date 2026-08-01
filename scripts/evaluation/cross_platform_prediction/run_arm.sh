#!/bin/bash
# run_arm.sh
# ==========
# ARM cross-platform experiments: cross-frequency, cross-processor,
# and cross-system LOOCV with top-K ablations.
# All at 10M, 100M, and 1000M granularity.
#
# A suite is skipped automatically (with a [SKIP] message) if the
# corresponding processed_data/<suite> folder doesn't exist for the data
# dir(s) involved, so this script adapts as new data is collected without
# needing hardcoded per-platform suite lists.
#
# Usage:
#   ./run_arm.sh [section]   (default: all)
#
# Sections:
#   arm_cf        ARM cross-frequency (desktop, server, edge)
#   arm_cp        ARM edge cross-processor
#   arm_cs        ARM cross-system
#   all           everything
#
# Override output root:
#   OUT_ROOT=/path  ./run_arm.sh
#
# Per-section suite/ablation rules:
#   Cross-frequency (SUITES_CF): dacapo_c1 and dacapo_c2 get a full run only
#     (no top-4/top-6 -- see cf_runner). spec_2017 and spec_2026 get full +
#     top-4 + top-6.
#   Cross-processor (SUITES_CP): dacapo_c1, spec_2017, spec_2026 all get
#     full + top-4 + top-6. No dacapo_c2.
#   Cross-system (SUITES_CS): dacapo_c1, spec_2017, spec_2026 get a full
#     run only (no top-4/top-6). No dacapo_c2.
# No suite gets an automatic instability-pruned twin here (cassandra,
# tradebeans, tradesoap, h2o, kafka dropped -- see
# shared_features.UNSTABLE_DACAPO_WORKLOADS); that's only run for the x86
# machines, in run_x86.sh's cross-frequency section.

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
CF_ARM="$SCRIPT_DIR/cross_frequency/cross_freq_arm.py"
CP_ARM="$SCRIPT_DIR/cross_processor/cross_proc_arm.py"
CS="$SCRIPT_DIR/cross_system/cross_sys_arm.py"
TOP_COUNTERS="$SCRIPT_DIR/top_counters.py"

TOP_K_VALUES=(4 6)
SUITES_CF=(dacapo_c1 dacapo_c2 spec_2017 spec_2026)
SUITES_CP=(dacapo_c1 spec_2017 spec_2026)
SUITES_CS=(dacapo_c1 spec_2017 spec_2026)

# ---------------------------------------------------------------------------
# Prediction target per machine.
#
# These values reproduce the released results and are NOT interchangeable.
# `ref_cycles` is a hardware counter only on x86. On the Arm edge board it is
# derived as cpu_cycles * (2.0 / freq) -- see derive_ref_cycles() in
# scripts/data_processing/arm_edge_heterogeneous/process_raw_data.py -- and on
# the Arm desktop and Arm server it is not collected at all, so those two
# machines can only use cpu_cycles.
#
# Consequence: cross-frequency MAPE is not directly comparable between the
# edge board and the desktop/server, because a copy baseline scores ~9% under
# cpu_cycles and ~66% under the derived ref_cycles. Override with
# TARGET_KEY=<key> to force one target everywhere (the desktop and server will
# then produce zero folds under ref_cycles).
# ---------------------------------------------------------------------------
declare -A TARGET_BY_MACHINE=(
    [arm_desktop]=cpu_cycles
    [arm_server]=cpu_cycles
    [arm_edge_heterogeneous]=ref_cycles
)
target_for() { echo "${TARGET_KEY:-${TARGET_BY_MACHINE[$1]}}"; }

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
#                        ARM CROSS-FREQUENCY
# ###########################################################################
if [[ "$TARGET" == "arm_cf" || "$TARGET" == "all" ]]; then

    # -----------------------------------------------------------------------
    # ARM DESKTOP (single CPU)
    # -----------------------------------------------------------------------
    for gran in 10M 100M 1000M; do
        data_dir="$REPO_ROOT/processed_data_${gran}/arm_desktop"

        for suite in "${SUITES_CF[@]}"; do
            if ! suite_dir_exists "$data_dir" "$suite"; then
                echo "  [SKIP] arm_desktop ($gran) | Cross-Freq | $suite: no data at $data_dir/$suite"
                continue
            fi
            echo ""; echo "=== arm_desktop ($gran) | Cross-Freq | $suite ==="
            "$(cf_runner "$suite")" "$OUT_ROOT/cross_freq/arm_desktop_${gran}/$suite/full" \
                "$PY" "$CF_ARM" --data_dir "$data_dir" --suite "$suite" --strict_loocv --target_key "$(target_for arm_desktop)"
        done
    done

    # -----------------------------------------------------------------------
    # ARM SERVER (single CPU)
    # -----------------------------------------------------------------------
    for gran in 10M 100M 1000M; do
        data_dir="$REPO_ROOT/processed_data_${gran}/arm_server"

        for suite in "${SUITES_CF[@]}"; do
            if ! suite_dir_exists "$data_dir" "$suite"; then
                echo "  [SKIP] arm_server ($gran) | Cross-Freq | $suite: no data at $data_dir/$suite"
                continue
            fi
            echo ""; echo "=== arm_server ($gran) | Cross-Freq | $suite ==="
            "$(cf_runner "$suite")" "$OUT_ROOT/cross_freq/arm_server_${gran}/$suite/full" \
                "$PY" "$CF_ARM" --data_dir "$data_dir" --suite "$suite" --strict_loocv --target_key "$(target_for arm_server)"
        done
    done

    # -----------------------------------------------------------------------
    # ARM EDGE HETEROGENEOUS — OoO cpu4 only (cpu1 InO has only 1GHz)
    # -----------------------------------------------------------------------
    for gran in 10M 100M 1000M; do
        data_dir="$REPO_ROOT/processed_data_${gran}/arm_edge_heterogeneous"

        for suite in "${SUITES_CF[@]}"; do
            if ! suite_dir_exists "$data_dir" "$suite"; then
                echo "  [SKIP] arm_edge ($gran) | Cross-Freq | OoO (cpu4) | $suite: no data at $data_dir/$suite"
                continue
            fi
            echo ""; echo "=== arm_edge ($gran) | Cross-Freq | OoO (cpu4) | $suite ==="
            "$(cf_runner "$suite")" "$OUT_ROOT/cross_freq/arm_edge_${gran}/cpu4/$suite/full" \
                "$PY" "$CF_ARM" --data_dir "$data_dir" --target_cpu 4 --suite "$suite" --strict_loocv --target_key "$(target_for arm_edge_heterogeneous)"
        done
    done

fi


# ###########################################################################
#                        ARM CROSS-PROCESSOR
# ###########################################################################
if [[ "$TARGET" == "arm_cp" || "$TARGET" == "all" ]]; then

    for gran in 10M 100M 1000M; do
        data_dir="$REPO_ROOT/processed_data_${gran}/arm_edge_heterogeneous"

        for suite in "${SUITES_CP[@]}"; do
            if ! suite_dir_exists "$data_dir" "$suite"; then
                echo "  [SKIP] arm_edge ($gran) | Cross-Proc | $suite: no data at $data_dir/$suite"
                continue
            fi

            echo ""; echo "=== arm_edge ($gran) | Cross-Proc | OoO → InO | $suite ==="
            run_with_ablation "$OUT_ROOT/cross_proc/arm_edge_${gran}/cpu4_to_cpu1/$suite/full" \
                "$PY" "$CP_ARM" --data_dir "$data_dir" --src_cpu 4 --tgt_cpu 1 --suite "$suite" --strict_loocv --target_key "$(target_for arm_edge_heterogeneous)"

            echo ""; echo "=== arm_edge ($gran) | Cross-Proc | InO → OoO | $suite ==="
            run_with_ablation "$OUT_ROOT/cross_proc/arm_edge_${gran}/cpu1_to_cpu4/$suite/full" \
                "$PY" "$CP_ARM" --data_dir "$data_dir" --src_cpu 1 --tgt_cpu 4 --suite "$suite" --strict_loocv --target_key "$(target_for arm_edge_heterogeneous)"
        done
    done

fi


# ###########################################################################
#                        ARM CROSS-SYSTEM
# ###########################################################################
# Restricted to matched 1.0GHz -> 1.0GHz for now (--freq 1.0) — cross-system
# already gives a 4x4 platform matrix per suite (server, desktop, edge InO,
# edge OoO); sweeping every source x target frequency combination on top of
# that is not needed yet. Drop --freq 1.0 below to re-enable the full
# frequency sweep. Being matched-frequency-only is also what makes edge InO
# (cpu1) includable here at all -- it only has a 1.0GHz collection, so it
# can't take part in the frequency-sweep sections (cross-freq, cross-proc).
# Edge InO <-> edge OoO itself is cross-processor territory (same chip,
# different core), not cross-system, so that pair is intentionally absent
# here -- see the ARM CROSS-PROCESSOR section.
if [[ "$TARGET" == "arm_cs" || "$TARGET" == "all" ]]; then

    for gran in 10M 100M 1000M; do
        ARM_SERVER="$REPO_ROOT/processed_data_${gran}/arm_server"
        ARM_DESKTOP="$REPO_ROOT/processed_data_${gran}/arm_desktop"
        ARM_EDGE="$REPO_ROOT/processed_data_${gran}/arm_edge_heterogeneous"
        OUT="$OUT_ROOT/cross_sys/arm_${gran}"

        # ---------------------------------------------------------------
        # ARM server <-> desktop
        # ---------------------------------------------------------------
        for suite in "${SUITES_CS[@]}"; do
            if ! suite_dir_exists "$ARM_SERVER" "$suite" || ! suite_dir_exists "$ARM_DESKTOP" "$suite"; then
                echo "  [SKIP] arm ($gran) | Cross-Sys | server<->desktop | $suite: missing in $ARM_SERVER or $ARM_DESKTOP"
                continue
            fi

            echo ""; echo "=== arm ($gran) | Cross-Sys | Server → Desktop | $suite ==="
            run_full_only "$OUT/server_to_desktop/$suite/full" \
                "$PY" "$CS" --src_data_dir "$ARM_SERVER" --tgt_data_dir "$ARM_DESKTOP" \
                    --src_label arm_server --tgt_label arm_desktop --suite "$suite" --freq 1.0

            echo ""; echo "=== arm ($gran) | Cross-Sys | Desktop → Server | $suite ==="
            run_full_only "$OUT/desktop_to_server/$suite/full" \
                "$PY" "$CS" --src_data_dir "$ARM_DESKTOP" --tgt_data_dir "$ARM_SERVER" \
                    --src_label arm_desktop --tgt_label arm_server --suite "$suite" --freq 1.0
        done

        # ---------------------------------------------------------------
        # ARM server <-> edge OoO
        # ---------------------------------------------------------------
        for suite in "${SUITES_CS[@]}"; do
            if ! suite_dir_exists "$ARM_SERVER" "$suite" || ! suite_dir_exists "$ARM_EDGE" "$suite"; then
                echo "  [SKIP] arm ($gran) | Cross-Sys | server<->edge_ooo | $suite: missing in $ARM_SERVER or $ARM_EDGE"
                continue
            fi

            echo ""; echo "=== arm ($gran) | Cross-Sys | Server → Edge OoO | $suite ==="
            run_full_only "$OUT/server_to_edge_ooo/$suite/full" \
                "$PY" "$CS" --src_data_dir "$ARM_SERVER" --tgt_data_dir "$ARM_EDGE" \
                    --tgt_cpu 4 --src_label arm_server --tgt_label arm_edge_ooo --suite "$suite" --freq 1.0

            echo ""; echo "=== arm ($gran) | Cross-Sys | Edge OoO → Server | $suite ==="
            run_full_only "$OUT/edge_ooo_to_server/$suite/full" \
                "$PY" "$CS" --src_data_dir "$ARM_EDGE" --tgt_data_dir "$ARM_SERVER" \
                    --src_cpu 4 --src_label arm_edge_ooo --tgt_label arm_server --suite "$suite" --freq 1.0
        done

        # ---------------------------------------------------------------
        # ARM desktop <-> edge OoO
        # ---------------------------------------------------------------
        for suite in "${SUITES_CS[@]}"; do
            if ! suite_dir_exists "$ARM_DESKTOP" "$suite" || ! suite_dir_exists "$ARM_EDGE" "$suite"; then
                echo "  [SKIP] arm ($gran) | Cross-Sys | desktop<->edge_ooo | $suite: missing in $ARM_DESKTOP or $ARM_EDGE"
                continue
            fi

            echo ""; echo "=== arm ($gran) | Cross-Sys | Desktop → Edge OoO | $suite ==="
            run_full_only "$OUT/desktop_to_edge_ooo/$suite/full" \
                "$PY" "$CS" --src_data_dir "$ARM_DESKTOP" --tgt_data_dir "$ARM_EDGE" \
                    --tgt_cpu 4 --src_label arm_desktop --tgt_label arm_edge_ooo --suite "$suite" --freq 1.0

            echo ""; echo "=== arm ($gran) | Cross-Sys | Edge OoO → Desktop | $suite ==="
            run_full_only "$OUT/edge_ooo_to_desktop/$suite/full" \
                "$PY" "$CS" --src_data_dir "$ARM_EDGE" --tgt_data_dir "$ARM_DESKTOP" \
                    --src_cpu 4 --src_label arm_edge_ooo --tgt_label arm_desktop --suite "$suite" --freq 1.0
        done

        # ---------------------------------------------------------------
        # ARM server <-> edge InO -- only possible for cross-system because
        # it's matched to 1.0GHz (cpu1's only frequency point); a full
        # frequency sweep isn't available for this core.
        # ---------------------------------------------------------------
        for suite in "${SUITES_CS[@]}"; do
            if ! suite_dir_exists "$ARM_SERVER" "$suite" || ! suite_dir_exists "$ARM_EDGE" "$suite"; then
                echo "  [SKIP] arm ($gran) | Cross-Sys | server<->edge_ino | $suite: missing in $ARM_SERVER or $ARM_EDGE"
                continue
            fi

            echo ""; echo "=== arm ($gran) | Cross-Sys | Server → Edge InO | $suite ==="
            run_full_only "$OUT/server_to_edge_ino/$suite/full" \
                "$PY" "$CS" --src_data_dir "$ARM_SERVER" --tgt_data_dir "$ARM_EDGE" \
                    --tgt_cpu 1 --src_label arm_server --tgt_label arm_edge_ino --suite "$suite" --freq 1.0

            echo ""; echo "=== arm ($gran) | Cross-Sys | Edge InO → Server | $suite ==="
            run_full_only "$OUT/edge_ino_to_server/$suite/full" \
                "$PY" "$CS" --src_data_dir "$ARM_EDGE" --tgt_data_dir "$ARM_SERVER" \
                    --src_cpu 1 --src_label arm_edge_ino --tgt_label arm_server --suite "$suite" --freq 1.0
        done

        # ---------------------------------------------------------------
        # ARM desktop <-> edge InO
        # ---------------------------------------------------------------
        for suite in "${SUITES_CS[@]}"; do
            if ! suite_dir_exists "$ARM_DESKTOP" "$suite" || ! suite_dir_exists "$ARM_EDGE" "$suite"; then
                echo "  [SKIP] arm ($gran) | Cross-Sys | desktop<->edge_ino | $suite: missing in $ARM_DESKTOP or $ARM_EDGE"
                continue
            fi

            echo ""; echo "=== arm ($gran) | Cross-Sys | Desktop → Edge InO | $suite ==="
            run_full_only "$OUT/desktop_to_edge_ino/$suite/full" \
                "$PY" "$CS" --src_data_dir "$ARM_DESKTOP" --tgt_data_dir "$ARM_EDGE" \
                    --tgt_cpu 1 --src_label arm_desktop --tgt_label arm_edge_ino --suite "$suite" --freq 1.0

            echo ""; echo "=== arm ($gran) | Cross-Sys | Edge InO → Desktop | $suite ==="
            run_full_only "$OUT/edge_ino_to_desktop/$suite/full" \
                "$PY" "$CS" --src_data_dir "$ARM_EDGE" --tgt_data_dir "$ARM_DESKTOP" \
                    --src_cpu 1 --src_label arm_edge_ino --tgt_label arm_desktop --suite "$suite" --freq 1.0
        done
    done

fi


echo ""; echo "All ARM models complete. Results in: $OUT_ROOT"

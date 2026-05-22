#!/usr/bin/env bash
# run_desktop_stall_experiment.sh
# ================================
# Compares cross-system prediction error for ARM desktop pairs at 1000M
# with and without ixb_stall_rate + sx_stall_rate features.
#
# Run from scripts/evaluation/cross_platform_prediction/cross_system/
#   bash run_desktop_stall_experiment.sh

set -euo pipefail

SCRIPT="python3 cross_sys_arm.py"
DATA="../../../../processed_data_1000M"
BASE_OUT="../../../../results/cross_platform/cross_system"

COMMON_ARGS="--freq 1.0 --suite spec --strict_loocv"
EXCLUDE_ARGS="--exclude_features ixb_stall_rate sx_stall_rate"

PAIRS=(
    "arm_desktop arm_server"
    "arm_server arm_desktop"
    "arm_desktop arm_edge_heterogeneous --tgt_cpu 1"
    "arm_edge_heterogeneous arm_desktop --src_cpu 1"
    "arm_desktop arm_edge_heterogeneous --tgt_cpu 4"
    "arm_edge_heterogeneous arm_desktop --src_cpu 4"
)

for pair in "${PAIRS[@]}"; do
    read -ra PARTS <<< "$pair"
    SRC="${PARTS[0]}"
    TGT="${PARTS[1]}"
    EXTRA_ARGS="${PARTS[*]:2}"   # e.g. "--tgt_cpu 1"

    # Build a compact label for the output directory
    EXTRA_LABEL=""
    if [[ "$EXTRA_ARGS" == *"--tgt_cpu 1"* ]]; then EXTRA_LABEL="_tgt_ino"; fi
    if [[ "$EXTRA_ARGS" == *"--tgt_cpu 4"* ]]; then EXTRA_LABEL="_tgt_ooo"; fi
    if [[ "$EXTRA_ARGS" == *"--src_cpu 1"* ]]; then EXTRA_LABEL="_src_ino"; fi
    if [[ "$EXTRA_ARGS" == *"--src_cpu 4"* ]]; then EXTRA_LABEL="_src_ooo"; fi

    BASE_DIR="${BASE_OUT}/${SRC}_to_${TGT}${EXTRA_LABEL}_1000M_spec"
    EXCL_DIR="${BASE_OUT}/${SRC}_to_${TGT}${EXTRA_LABEL}_1000M_spec_no_stall"

    echo ""
    echo "========================================================"
    echo "  Baseline : $SRC -> $TGT${EXTRA_LABEL}"
    echo "========================================================"
    # shellcheck disable=SC2086
    $SCRIPT \
        --src_data_dir "$DATA/$SRC/" \
        --tgt_data_dir "$DATA/$TGT/" \
        --out_dir      "$BASE_DIR"   \
        $COMMON_ARGS $EXTRA_ARGS

    echo ""
    echo "========================================================"
    echo "  No-stall : $SRC -> $TGT${EXTRA_LABEL}"
    echo "========================================================"
    # shellcheck disable=SC2086
    $SCRIPT \
        --src_data_dir "$DATA/$SRC/" \
        --tgt_data_dir "$DATA/$TGT/" \
        --out_dir      "$EXCL_DIR"   \
        $COMMON_ARGS $EXTRA_ARGS $EXCLUDE_ARGS
done

echo ""
echo "========================================================"
echo "  Experiment complete. Collecting results…"
echo "========================================================"
python3 ../../../../scripts/data_processing/collect_cross_system_results.py

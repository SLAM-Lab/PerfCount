#!/bin/bash
# run_arm_edge.sh — launch the scheduling / DVFS simulator for RB5 (arm_edge_heterogeneous)
#
# Usage:
#   ./run_arm_edge.sh                                          # run all policies
#   ./run_arm_edge.sh --list-policies                          # print every policy name
#   ./run_arm_edge.sh --policies Ondemand_OOO EAS_Hetero ...  # specific policies
#
# INPUT_DIR / OUTPUT_DIR can be overridden via environment variables:
#   INPUT_DIR=/my/data OUTPUT_DIR=/my/out ./run_arm_edge.sh
#
# Generate the arm_edge speedup CSVs first:
#   python3 utils/generate_speedup_matrix.py \
#       --input_dir  processed_data_10M/arm_edge_heterogeneous \
#       --output_dir results/scheduling/arm_edge_10M \
#       --platform   arm_edge

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

INPUT_DIR="${INPUT_DIR:-$REPO_ROOT/results/scheduling/arm_edge_10M/granular_phase_traces}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/scheduling/arm_edge_output}"

python3 "$SCRIPT_DIR/../src/main.py" \
    --platform   arm_edge \
    --input_dir  "$INPUT_DIR"  \
    --output_dir "$OUTPUT_DIR" \
    "$@"

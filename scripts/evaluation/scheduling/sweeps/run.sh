#!/bin/bash
# run.sh — launch the scheduling / DVFS simulator
#
# Usage:
#   ./run.sh                                          # run all policies
#   ./run.sh --list-policies                          # print every policy name
#   ./run.sh --policies Ondemand_P Intel_HWP_P ...   # specific policies
#
# INPUT_DIR / OUTPUT_DIR can be overridden via environment variables:
#   INPUT_DIR=/my/data OUTPUT_DIR=/my/out ./run.sh --policies Ondemand_P

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

INPUT_DIR="${INPUT_DIR:-$REPO_ROOT/results/scheduling/granular_phase_traces_10M}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/scheduling/output}"

python3 "$SCRIPT_DIR/../src/main.py" \
    --input_dir  "$INPUT_DIR"  \
    --output_dir "$OUTPUT_DIR" \
    "$@"

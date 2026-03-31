#!/bin/bash

# Wrapper script for process_log_results.py
# Usage: ./process_log_results.sh --input <dir> --output <dir>
#   Or:  ./process_log_results.sh <input_dir> [output_dir]

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# If called with --input and --output, pass through
if [[ "$1" == "--input" ]] || [[ "$1" == "-i" ]]; then
    python3 "${SCRIPT_DIR}/process_log_results.py" "$@"
    exit $?
fi

# Simple usage: ./process_log_results.sh <input_dir> [output_dir]
if [ $# -lt 1 ]; then
    echo "Usage: $0 <input_directory> [output_directory]"
    echo ""
    echo "Examples:"
    echo "  $0 counter_forecasting_logs/BUS_CYCLES"
    echo "  $0 counter_forecasting_logs/BUS_CYCLES results/BUS_CYCLES"
    echo ""
    echo "Or use full syntax:"
    echo "  $0 --input <dir> --output <dir>"
    exit 1
fi

INPUT_DIR="$1"
OUTPUT_DIR="${2:-$INPUT_DIR}"

python3 "${SCRIPT_DIR}/process_log_results.py" --input "$INPUT_DIR" --output "$OUTPUT_DIR"


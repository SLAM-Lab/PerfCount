#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

RAW_DATA="$ROOT_DIR/raw_data"
OUT_DATA="$ROOT_DIR/processed_data_10M"

ALL_MACHINES=(x86_server x86_desktop_heterogeneous arm_server arm_desktop arm_edge_heterogeneous)

declare -A ARCH_MAP=(
    [x86_server]=x86
    [x86_desktop_heterogeneous]=x86
    [arm_server]=arm
    [arm_desktop]=arm
    [arm_edge_heterogeneous]=arm
)

usage() {
    cat <<'EOF'
Usage: process_all.sh [OPTIONS]

Options:
  --machines M1,M2,...   Machines to process (default: all)
  --suites   S1,S2,...   Benchmark suites to process (default: all)
  --freqs    F1,F2,...   Frequencies to process (default: all)
  --jobs     N           Number of parallel workers (default: nproc)
  --compact-only         Skip processing, only run compaction on existing 10M data
  --list                 Show available machines and suites, then exit
  -h, --help             Show this help

Available machines:
  x86_server, x86_desktop_heterogeneous,
  arm_server, arm_desktop, arm_edge_heterogeneous

Examples:
  process_all.sh                                         # everything
  process_all.sh --machines x86_server                   # one machine
  process_all.sh --suites spec_2026,dacapo_c2            # specific suites
  process_all.sh --machines arm_server --freqs 1.0,2.0   # specific freqs
  process_all.sh --list                                  # show what's available
EOF
}

MACHINES=()
SUITES=()
FREQS=""
JOBS=$(nproc)
LIST=false
COMPACT_ONLY=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --machines) IFS=',' read -ra MACHINES <<< "$2"; shift 2 ;;
        --suites)   IFS=',' read -ra SUITES <<< "$2"; shift 2 ;;
        --freqs)    FREQS="$2"; shift 2 ;;
        --jobs)     JOBS="$2"; shift 2 ;;
        --compact-only) COMPACT_ONLY=true; shift ;;
        --list)     LIST=true; shift ;;
        -h|--help)  usage; exit 0 ;;
        *)          echo "Unknown option: $1"; usage; exit 1 ;;
    esac
done

if [[ ${#MACHINES[@]} -eq 0 ]]; then
    MACHINES=("${ALL_MACHINES[@]}")
fi

# Validate machine names
for machine in "${MACHINES[@]}"; do
    if [[ -z "${ARCH_MAP[$machine]+x}" ]]; then
        echo "Error: unknown machine '$machine'"
        echo "Available: ${ALL_MACHINES[*]}"
        exit 1
    fi
done

if $LIST; then
    for machine in "${MACHINES[@]}"; do
        echo "$machine (${ARCH_MAP[$machine]}):"
        machine_raw="$RAW_DATA/$machine"
        if [[ -d "$machine_raw" ]]; then
            for suite_dir in "$machine_raw"/*/; do
                [[ -d "$suite_dir" ]] || continue
                suite=$(basename "$suite_dir")
                count=$(find "$suite_dir" -name "*.out" | wc -l)
                freqs=$(find "$suite_dir" -name "*.out" -printf '%f\n' \
                    | grep -oP '[\d.]+GHz' | sort -u | tr '\n' ' ')
                echo "  $suite  ($count files)  freqs: $freqs"
            done
        else
            echo "  (no raw_data directory found)"
        fi
        echo ""
    done
    exit 0
fi

if ! $COMPACT_ONLY; then
for machine in "${MACHINES[@]}"; do
    arch="${ARCH_MAP[$machine]}"
    machine_raw="$RAW_DATA/$machine"
    machine_out="$OUT_DATA/$machine"
    script="$SCRIPT_DIR/$machine/process_raw_data.py"

    if [[ ! -d "$machine_raw" ]]; then
        echo "Warning: $machine_raw does not exist, skipping $machine"
        continue
    fi

    freq_args=()
    if [[ -n "$FREQS" ]]; then
        freq_args=(--freqs "$FREQS")
    fi

    if [[ ${#SUITES[@]} -gt 0 ]]; then
        for suite in "${SUITES[@]}"; do
            suite_raw="$machine_raw/$suite"
            if [[ ! -d "$suite_raw" ]]; then
                continue
            fi
            suite_out="$machine_out/$suite"
            echo "=== Processing $machine / $suite ==="
            python3 "$script" \
                --raw_dir "$suite_raw" \
                --out_dir "$suite_out" \
                --arch "$arch" \
                --jobs "$JOBS" \
                "${freq_args[@]+"${freq_args[@]}"}"
            echo ""
        done
    else
        echo "=== Processing $machine ==="
        python3 "$script" \
            --raw_dir "$machine_raw" \
            --out_dir "$machine_out" \
            --arch "$arch" \
            --jobs "$JOBS" \
            "${freq_args[@]+"${freq_args[@]}"}"
        echo ""
    fi
done
fi

echo "=== Generating compacted data ==="

COMPACT_SCRIPT="$SCRIPT_DIR/utils/compact_data.py"
OUT_100M="$ROOT_DIR/processed_data_100M"
OUT_1000M="$ROOT_DIR/processed_data_1000M"

for machine in "${MACHINES[@]}"; do
    machine_10m="$OUT_DATA/$machine"
    if [[ ! -d "$machine_10m" ]]; then
        continue
    fi

    echo "=== Compacting $machine -> 100M ==="
    python3 "$COMPACT_SCRIPT" \
        --in_dir "$machine_10m" \
        --out_dir "$OUT_100M/$machine" \
        --block_size 10

    echo "=== Compacting $machine -> 1000M ==="
    python3 "$COMPACT_SCRIPT" \
        --in_dir "$machine_10m" \
        --out_dir "$OUT_1000M/$machine" \
        --block_size 100

    echo ""
done

echo "=== Done ==="

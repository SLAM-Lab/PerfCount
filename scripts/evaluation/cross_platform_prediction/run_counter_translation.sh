#!/bin/bash
# run_counter_translation.sh
# ==========================
# Per-counter cpu16 -> cpu0 translation models for x86 heterogeneous platforms.
#
# Trains one CatBoost model per (target counter, suite, freq-pair) combination.
# The first counter in --input_counters is the prediction target; the rest
# (plus ALWAYS_KEEP: instructions, cpu_cycles, ref_cycles) are the features.
#
# Top-2 and top-4 counters derived from DT feature importance aggregation
# over all cross_proc DT runs (aggregate_feature_importance.py).
#
# Usage:
#   ./run_counter_translation.sh [x86|all]   (default: all)
#
# Override data/output roots:
#   DATA_10M=/path  OUT_ROOT=/path  ./run_counter_translation.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
TARGET="${1:-all}"

DATA_10M="${DATA_10M:-$REPO_ROOT/processed_data_10M}"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/results/cross_platform}"

CT_X86="$SCRIPT_DIR/cross_processor/cross_proc_counter_translation.py"
X86_DATA="$DATA_10M/x86_desktop_heterogeneous"
X86_OUT="$OUT_ROOT/cross_proc/x86_10M/counter_translation/cpu16_to_cpu0"

# Top-2 and top-4 counters by DT feature importance (excluding instructions/ref_cycles).
# cpu_cycles is always kept by shared_features.py regardless.
X86_TOP2="cpu_cycles branch_load_misses"
X86_TOP4="cpu_cycles branch_load_misses branch_misses dtlb_loads"

# All counters we build translation models for (top-4 + instructions + ref_cycles).
# Each is run as the first argument so it becomes the prediction target.
TARGETS="cpu_cycles branch_load_misses branch_misses dtlb_loads instructions ref_cycles"

if [[ "$TARGET" == "x86" || "$TARGET" == "all" ]]; then

    for suite in spec2017 spec2026 dacapo; do
        for counter in $TARGETS; do

            echo ""
            echo "=== x86 | cpu16 -> cpu0 | $suite | target: $counter | top2 ==="
            python3 $CT_X86 \
                --data_dir $X86_DATA \
                --src_cpu 16 --tgt_cpu 0 \
                --suite $suite --strict_loocv \
                --input_counters $counter $X86_TOP2 \
                --out_dir $X86_OUT/$suite/$counter/top2

            echo ""
            echo "=== x86 | cpu16 -> cpu0 | $suite | target: $counter | top4 ==="
            python3 $CT_X86 \
                --data_dir $X86_DATA \
                --src_cpu 16 --tgt_cpu 0 \
                --suite $suite --strict_loocv \
                --input_counters $counter $X86_TOP4 \
                --out_dir $X86_OUT/$suite/$counter/top4

        done
    done

fi

echo ""
echo "All counter translation models complete. Results in: $X86_OUT"

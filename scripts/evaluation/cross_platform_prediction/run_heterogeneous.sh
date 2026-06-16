#!/bin/bash
# run_all_models.sh
# =================
# Runs all cross-frequency and cross-processor CatBoost models for x86 and ARM.
# Each model is run with full counters. Top-2 and top-4 variants are commented out
# pending re-running the ablation studies to confirm the best counters per suite.
# x86 suites are run independently: spec2017, spec2026, dacapo.
# ARM suites are run independently: spec, dacapo.
#
# Usage:
#   ./run_all_models.sh [x86|arm_edge|all]   (default: all)
#
# Override data/output roots:
#   DATA_10M=/path  OUT_ROOT=/path  ./run_all_models.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
TARGET="${1:-all}"

DATA_10M="${DATA_10M:-$REPO_ROOT/processed_data_10M}"
DATA_100M="${DATA_100M:-$REPO_ROOT/processed_data_100M}"
DATA_1000M="${DATA_1000M:-$REPO_ROOT/processed_data_1000M}"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/results/cross_platform}"

# Script paths
CF_X86="$SCRIPT_DIR/cross_frequency/cross_freq_x86.py"
CF_ARM="$SCRIPT_DIR/cross_frequency/cross_freq_arm.py"
CP_X86="$SCRIPT_DIR/cross_processor/cross_proc_x86.py"
CP_ARM="$SCRIPT_DIR/cross_processor/cross_proc_arm.py"

# Data directories
X86_DATA="$DATA_10M/x86_desktop_heterogeneous"
X86_DATA_100M="$DATA_100M/x86_desktop_heterogeneous"
X86_DATA_1000M="$DATA_1000M/x86_desktop_heterogeneous"
ARM_DATA="$DATA_10M/arm_edge_heterogeneous"
ARM_DATA_100M="$DATA_100M/arm_edge_heterogeneous"
ARM_DATA_1000M="$DATA_1000M/arm_edge_heterogeneous"

# Output directories
X86_CF="$OUT_ROOT/cross_freq/x86_10M"
X86_CP="$OUT_ROOT/cross_proc/x86_10M"
X86_CP_100M="$OUT_ROOT/cross_proc/x86_100M"
X86_CP_1000M="$OUT_ROOT/cross_proc/x86_1000M"
ARM_CF="$OUT_ROOT/cross_freq/arm_edge_10M"
ARM_CP="$OUT_ROOT/cross_proc/arm_edge_10M"
ARM_CP_100M="$OUT_ROOT/cross_proc/arm_edge_100M"
ARM_CP_1000M="$OUT_ROOT/cross_proc/arm_edge_1000M"

# Top counters
# Cross-frequency top counters (per-core, derived from full-run feature importances)
X86_CF_CPU0_TOP2="cache_misses llc_misses"
X86_CF_CPU0_TOP4="cache_misses llc_misses branch_misses branches"
X86_CF_CPU16_TOP2="cache_misses branch_misses"
X86_CF_CPU16_TOP4="cache_misses branch_misses dtlb_load_misses llc_misses"

# Cross-processor top counters (derived from cpu0_to_cpu16 full-run feature importances)
X86_CP_TOP2="branch_misses l1_icache_load_misses"
X86_CP_TOP4="branch_misses l1_icache_load_misses l1_dcache_load_misses branches"
X86_CP_MATCHED_TOP2="branch_misses branches"
X86_CP_MATCHED_TOP4="branch_misses branches l1_dcache_load_misses l1_icache_load_misses"

ARM_TOP2="l3d_cache_refill branch_misses"
ARM_TOP4="l3d_cache_refill branch_misses l2d_cache_refill cache_misses"


# ===========================================================================
# X86 DESKTOP — CROSS-FREQUENCY — P-CORE (cpu0)
# ===========================================================================
if [[ "$TARGET" == "x86" || "$TARGET" == "all" ]]; then

    echo ""; echo "=== x86 | Cross-Frequency | P-Core (cpu0) | SPEC 2017 ==="
    python3 $CF_X86 --data_dir $X86_DATA --target_cpu 0 --suite spec2017 --strict_loocv --out_dir $X86_CF/cpu0/spec2017/full
    python3 $CF_X86 --data_dir $X86_DATA --target_cpu 0 --suite spec2017 --strict_loocv --out_dir $X86_CF/cpu0/spec2017/top2 --input_counters $X86_CF_CPU0_TOP2
    python3 $CF_X86 --data_dir $X86_DATA --target_cpu 0 --suite spec2017 --strict_loocv --out_dir $X86_CF/cpu0/spec2017/top4 --input_counters $X86_CF_CPU0_TOP4

    echo ""; echo "=== x86 | Cross-Frequency | P-Core (cpu0) | SPEC 2026 ==="
    python3 $CF_X86 --data_dir $X86_DATA --target_cpu 0 --suite spec2026 --strict_loocv --out_dir $X86_CF/cpu0/spec2026/full
    python3 $CF_X86 --data_dir $X86_DATA --target_cpu 0 --suite spec2026 --strict_loocv --out_dir $X86_CF/cpu0/spec2026/top2 --input_counters $X86_CF_CPU0_TOP2
    python3 $CF_X86 --data_dir $X86_DATA --target_cpu 0 --suite spec2026 --strict_loocv --out_dir $X86_CF/cpu0/spec2026/top4 --input_counters $X86_CF_CPU0_TOP4

    echo ""; echo "=== x86 | Cross-Frequency | P-Core (cpu0) | DaCapo ==="
    python3 $CF_X86 --data_dir $X86_DATA --target_cpu 0 --suite dacapo --strict_loocv --out_dir $X86_CF/cpu0/dacapo/full
    python3 $CF_X86 --data_dir $X86_DATA --target_cpu 0 --suite dacapo --strict_loocv --out_dir $X86_CF/cpu0/dacapo/top2 --input_counters $X86_CF_CPU0_TOP2
    python3 $CF_X86 --data_dir $X86_DATA --target_cpu 0 --suite dacapo --strict_loocv --out_dir $X86_CF/cpu0/dacapo/top4 --input_counters $X86_CF_CPU0_TOP4

    # ===========================================================================
    # X86 DESKTOP — CROSS-FREQUENCY — E-CORE (cpu16)
    # ===========================================================================
    echo ""; echo "=== x86 | Cross-Frequency | E-Core (cpu16) | SPEC 2017 ==="
    python3 $CF_X86 --data_dir $X86_DATA --target_cpu 16 --suite spec2017 --strict_loocv --out_dir $X86_CF/cpu16/spec2017/full
    python3 $CF_X86 --data_dir $X86_DATA --target_cpu 16 --suite spec2017 --strict_loocv --out_dir $X86_CF/cpu16/spec2017/top2 --input_counters $X86_CF_CPU16_TOP2
    python3 $CF_X86 --data_dir $X86_DATA --target_cpu 16 --suite spec2017 --strict_loocv --out_dir $X86_CF/cpu16/spec2017/top4 --input_counters $X86_CF_CPU16_TOP4

    echo ""; echo "=== x86 | Cross-Frequency | E-Core (cpu16) | SPEC 2026 ==="
    python3 $CF_X86 --data_dir $X86_DATA --target_cpu 16 --suite spec2026 --strict_loocv --out_dir $X86_CF/cpu16/spec2026/full
    python3 $CF_X86 --data_dir $X86_DATA --target_cpu 16 --suite spec2026 --strict_loocv --out_dir $X86_CF/cpu16/spec2026/top2 --input_counters $X86_CF_CPU16_TOP2
    python3 $CF_X86 --data_dir $X86_DATA --target_cpu 16 --suite spec2026 --strict_loocv --out_dir $X86_CF/cpu16/spec2026/top4 --input_counters $X86_CF_CPU16_TOP4

    echo ""; echo "=== x86 | Cross-Frequency | E-Core (cpu16) | DaCapo ==="
    python3 $CF_X86 --data_dir $X86_DATA --target_cpu 16 --suite dacapo --strict_loocv --out_dir $X86_CF/cpu16/dacapo/full
    python3 $CF_X86 --data_dir $X86_DATA --target_cpu 16 --suite dacapo --strict_loocv --out_dir $X86_CF/cpu16/dacapo/top2 --input_counters $X86_CF_CPU16_TOP2
    python3 $CF_X86 --data_dir $X86_DATA --target_cpu 16 --suite dacapo --strict_loocv --out_dir $X86_CF/cpu16/dacapo/top4 --input_counters $X86_CF_CPU16_TOP4

    # ===========================================================================
    # X86 DESKTOP — CROSS-PROCESSOR — P-Core → E-Core
    # ===========================================================================
    echo ""; echo "=== x86 | Cross-Processor | P-Core → E-Core | SPEC 2017 ==="
    python3 $CP_X86 --data_dir $X86_DATA --src_cpu 0 --tgt_cpu 16 --suite spec2017 --strict_loocv --out_dir $X86_CP/cpu0_to_cpu16/spec2017/full
    python3 $CP_X86 --data_dir $X86_DATA --src_cpu 0 --tgt_cpu 16 --suite spec2017 --strict_loocv --out_dir $X86_CP/cpu0_to_cpu16/spec2017/top2 --input_counters $X86_CP_TOP2
    python3 $CP_X86 --data_dir $X86_DATA --src_cpu 0 --tgt_cpu 16 --suite spec2017 --strict_loocv --out_dir $X86_CP/cpu0_to_cpu16/spec2017/top4 --input_counters $X86_CP_TOP4
    python3 $CP_X86 --data_dir $X86_DATA --src_cpu 0 --tgt_cpu 16 --suite spec2017 --strict_loocv --out_dir $X86_CP/cpu0_to_cpu16/spec2017/top2_matched --input_counters $X86_CP_MATCHED_TOP2
    python3 $CP_X86 --data_dir $X86_DATA --src_cpu 0 --tgt_cpu 16 --suite spec2017 --strict_loocv --out_dir $X86_CP/cpu0_to_cpu16/spec2017/top4_matched --input_counters $X86_CP_MATCHED_TOP4

    echo ""; echo "=== x86 | Cross-Processor | P-Core → E-Core | SPEC 2026 ==="
    python3 $CP_X86 --data_dir $X86_DATA --src_cpu 0 --tgt_cpu 16 --suite spec2026 --strict_loocv --out_dir $X86_CP/cpu0_to_cpu16/spec2026/full
    python3 $CP_X86 --data_dir $X86_DATA --src_cpu 0 --tgt_cpu 16 --suite spec2026 --strict_loocv --out_dir $X86_CP/cpu0_to_cpu16/spec2026/top2 --input_counters $X86_CP_TOP2
    python3 $CP_X86 --data_dir $X86_DATA --src_cpu 0 --tgt_cpu 16 --suite spec2026 --strict_loocv --out_dir $X86_CP/cpu0_to_cpu16/spec2026/top4 --input_counters $X86_CP_TOP4
    python3 $CP_X86 --data_dir $X86_DATA --src_cpu 0 --tgt_cpu 16 --suite spec2026 --strict_loocv --out_dir $X86_CP/cpu0_to_cpu16/spec2026/top2_matched --input_counters $X86_CP_MATCHED_TOP2
    python3 $CP_X86 --data_dir $X86_DATA --src_cpu 0 --tgt_cpu 16 --suite spec2026 --strict_loocv --out_dir $X86_CP/cpu0_to_cpu16/spec2026/top4_matched --input_counters $X86_CP_MATCHED_TOP4

    echo ""; echo "=== x86 | Cross-Processor | P-Core → E-Core | DaCapo ==="
    python3 $CP_X86 --data_dir $X86_DATA --src_cpu 0 --tgt_cpu 16 --suite dacapo --strict_loocv --out_dir $X86_CP/cpu0_to_cpu16/dacapo/full
    python3 $CP_X86 --data_dir $X86_DATA --src_cpu 0 --tgt_cpu 16 --suite dacapo --strict_loocv --out_dir $X86_CP/cpu0_to_cpu16/dacapo/top2 --input_counters $X86_CP_TOP2
    python3 $CP_X86 --data_dir $X86_DATA --src_cpu 0 --tgt_cpu 16 --suite dacapo --strict_loocv --out_dir $X86_CP/cpu0_to_cpu16/dacapo/top4 --input_counters $X86_CP_TOP4
    python3 $CP_X86 --data_dir $X86_DATA --src_cpu 0 --tgt_cpu 16 --suite dacapo --strict_loocv --out_dir $X86_CP/cpu0_to_cpu16/dacapo/top2_matched --input_counters $X86_CP_MATCHED_TOP2
    python3 $CP_X86 --data_dir $X86_DATA --src_cpu 0 --tgt_cpu 16 --suite dacapo --strict_loocv --out_dir $X86_CP/cpu0_to_cpu16/dacapo/top4_matched --input_counters $X86_CP_MATCHED_TOP4

    # ===========================================================================
    # X86 DESKTOP — CROSS-PROCESSOR — E-Core → P-Core
    # ===========================================================================
    echo ""; echo "=== x86 | Cross-Processor | E-Core → P-Core | SPEC 2017 ==="
    python3 $CP_X86 --data_dir $X86_DATA --src_cpu 16 --tgt_cpu 0 --suite spec2017 --strict_loocv --out_dir $X86_CP/cpu16_to_cpu0/spec2017/full
    python3 $CP_X86 --data_dir $X86_DATA --src_cpu 16 --tgt_cpu 0 --suite spec2017 --strict_loocv --out_dir $X86_CP/cpu16_to_cpu0/spec2017/top2 --input_counters $X86_CP_TOP2
    python3 $CP_X86 --data_dir $X86_DATA --src_cpu 16 --tgt_cpu 0 --suite spec2017 --strict_loocv --out_dir $X86_CP/cpu16_to_cpu0/spec2017/top4 --input_counters $X86_CP_TOP4

    echo ""; echo "=== x86 | Cross-Processor | E-Core → P-Core | SPEC 2026 ==="
    python3 $CP_X86 --data_dir $X86_DATA --src_cpu 16 --tgt_cpu 0 --suite spec2026 --strict_loocv --out_dir $X86_CP/cpu16_to_cpu0/spec2026/full
    python3 $CP_X86 --data_dir $X86_DATA --src_cpu 16 --tgt_cpu 0 --suite spec2026 --strict_loocv --out_dir $X86_CP/cpu16_to_cpu0/spec2026/top2 --input_counters $X86_CP_TOP2
    python3 $CP_X86 --data_dir $X86_DATA --src_cpu 16 --tgt_cpu 0 --suite spec2026 --strict_loocv --out_dir $X86_CP/cpu16_to_cpu0/spec2026/top4 --input_counters $X86_CP_TOP4

    echo ""; echo "=== x86 | Cross-Processor | E-Core → P-Core | DaCapo ==="
    python3 $CP_X86 --data_dir $X86_DATA --src_cpu 16 --tgt_cpu 0 --suite dacapo --strict_loocv --out_dir $X86_CP/cpu16_to_cpu0/dacapo/full
    python3 $CP_X86 --data_dir $X86_DATA --src_cpu 16 --tgt_cpu 0 --suite dacapo --strict_loocv --out_dir $X86_CP/cpu16_to_cpu0/dacapo/top2 --input_counters $X86_CP_TOP2
    python3 $CP_X86 --data_dir $X86_DATA --src_cpu 16 --tgt_cpu 0 --suite dacapo --strict_loocv --out_dir $X86_CP/cpu16_to_cpu0/dacapo/top4 --input_counters $X86_CP_TOP4

    # ===========================================================================
    # X86 DESKTOP — CROSS-PROCESSOR — P-Core → E-Core (100M)
    # ===========================================================================
    echo ""; echo "=== x86 | Cross-Processor | P-Core → E-Core | SPEC 2017 (100M) ==="
    python3 $CP_X86 --data_dir $X86_DATA_100M --src_cpu 0 --tgt_cpu 16 --suite spec2017 --strict_loocv --out_dir $X86_CP_100M/cpu0_to_cpu16/spec2017/full
    python3 $CP_X86 --data_dir $X86_DATA_100M --src_cpu 0 --tgt_cpu 16 --suite spec2017 --strict_loocv --out_dir $X86_CP_100M/cpu0_to_cpu16/spec2017/top2 --input_counters $X86_CP_TOP2
    python3 $CP_X86 --data_dir $X86_DATA_100M --src_cpu 0 --tgt_cpu 16 --suite spec2017 --strict_loocv --out_dir $X86_CP_100M/cpu0_to_cpu16/spec2017/top4 --input_counters $X86_CP_TOP4

    echo ""; echo "=== x86 | Cross-Processor | P-Core → E-Core | SPEC 2026 (100M) ==="
    python3 $CP_X86 --data_dir $X86_DATA_100M --src_cpu 0 --tgt_cpu 16 --suite spec2026 --strict_loocv --out_dir $X86_CP_100M/cpu0_to_cpu16/spec2026/full
    python3 $CP_X86 --data_dir $X86_DATA_100M --src_cpu 0 --tgt_cpu 16 --suite spec2026 --strict_loocv --out_dir $X86_CP_100M/cpu0_to_cpu16/spec2026/top2 --input_counters $X86_CP_TOP2
    python3 $CP_X86 --data_dir $X86_DATA_100M --src_cpu 0 --tgt_cpu 16 --suite spec2026 --strict_loocv --out_dir $X86_CP_100M/cpu0_to_cpu16/spec2026/top4 --input_counters $X86_CP_TOP4

    echo ""; echo "=== x86 | Cross-Processor | P-Core → E-Core | DaCapo (100M) ==="
    python3 $CP_X86 --data_dir $X86_DATA_100M --src_cpu 0 --tgt_cpu 16 --suite dacapo --strict_loocv --out_dir $X86_CP_100M/cpu0_to_cpu16/dacapo/full
    python3 $CP_X86 --data_dir $X86_DATA_100M --src_cpu 0 --tgt_cpu 16 --suite dacapo --strict_loocv --out_dir $X86_CP_100M/cpu0_to_cpu16/dacapo/top2 --input_counters $X86_CP_TOP2
    python3 $CP_X86 --data_dir $X86_DATA_100M --src_cpu 0 --tgt_cpu 16 --suite dacapo --strict_loocv --out_dir $X86_CP_100M/cpu0_to_cpu16/dacapo/top4 --input_counters $X86_CP_TOP4

    # ===========================================================================
    # X86 DESKTOP — CROSS-PROCESSOR — E-Core → P-Core (100M)
    # ===========================================================================
    echo ""; echo "=== x86 | Cross-Processor | E-Core → P-Core | SPEC 2017 (100M) ==="
    python3 $CP_X86 --data_dir $X86_DATA_100M --src_cpu 16 --tgt_cpu 0 --suite spec2017 --strict_loocv --out_dir $X86_CP_100M/cpu16_to_cpu0/spec2017/full
    python3 $CP_X86 --data_dir $X86_DATA_100M --src_cpu 16 --tgt_cpu 0 --suite spec2017 --strict_loocv --out_dir $X86_CP_100M/cpu16_to_cpu0/spec2017/top2 --input_counters $X86_CP_TOP2
    python3 $CP_X86 --data_dir $X86_DATA_100M --src_cpu 16 --tgt_cpu 0 --suite spec2017 --strict_loocv --out_dir $X86_CP_100M/cpu16_to_cpu0/spec2017/top4 --input_counters $X86_CP_TOP4

    echo ""; echo "=== x86 | Cross-Processor | E-Core → P-Core | SPEC 2026 (100M) ==="
    python3 $CP_X86 --data_dir $X86_DATA_100M --src_cpu 16 --tgt_cpu 0 --suite spec2026 --strict_loocv --out_dir $X86_CP_100M/cpu16_to_cpu0/spec2026/full
    python3 $CP_X86 --data_dir $X86_DATA_100M --src_cpu 16 --tgt_cpu 0 --suite spec2026 --strict_loocv --out_dir $X86_CP_100M/cpu16_to_cpu0/spec2026/top2 --input_counters $X86_CP_TOP2
    python3 $CP_X86 --data_dir $X86_DATA_100M --src_cpu 16 --tgt_cpu 0 --suite spec2026 --strict_loocv --out_dir $X86_CP_100M/cpu16_to_cpu0/spec2026/top4 --input_counters $X86_CP_TOP4

    echo ""; echo "=== x86 | Cross-Processor | E-Core → P-Core | DaCapo (100M) ==="
    python3 $CP_X86 --data_dir $X86_DATA_100M --src_cpu 16 --tgt_cpu 0 --suite dacapo --strict_loocv --out_dir $X86_CP_100M/cpu16_to_cpu0/dacapo/full
    python3 $CP_X86 --data_dir $X86_DATA_100M --src_cpu 16 --tgt_cpu 0 --suite dacapo --strict_loocv --out_dir $X86_CP_100M/cpu16_to_cpu0/dacapo/top2 --input_counters $X86_CP_TOP2
    python3 $CP_X86 --data_dir $X86_DATA_100M --src_cpu 16 --tgt_cpu 0 --suite dacapo --strict_loocv --out_dir $X86_CP_100M/cpu16_to_cpu0/dacapo/top4 --input_counters $X86_CP_TOP4

    # ===========================================================================
    # X86 DESKTOP — CROSS-PROCESSOR — P-Core → E-Core (1000M)
    # ===========================================================================
    echo ""; echo "=== x86 | Cross-Processor | P-Core → E-Core | SPEC 2017 (1000M) ==="
    python3 $CP_X86 --data_dir $X86_DATA_1000M --src_cpu 0 --tgt_cpu 16 --suite spec2017 --strict_loocv --out_dir $X86_CP_1000M/cpu0_to_cpu16/spec2017/full
    python3 $CP_X86 --data_dir $X86_DATA_1000M --src_cpu 0 --tgt_cpu 16 --suite spec2017 --strict_loocv --out_dir $X86_CP_1000M/cpu0_to_cpu16/spec2017/top2 --input_counters $X86_CP_TOP2
    python3 $CP_X86 --data_dir $X86_DATA_1000M --src_cpu 0 --tgt_cpu 16 --suite spec2017 --strict_loocv --out_dir $X86_CP_1000M/cpu0_to_cpu16/spec2017/top4 --input_counters $X86_CP_TOP4

    echo ""; echo "=== x86 | Cross-Processor | P-Core → E-Core | SPEC 2026 (1000M) ==="
    python3 $CP_X86 --data_dir $X86_DATA_1000M --src_cpu 0 --tgt_cpu 16 --suite spec2026 --strict_loocv --out_dir $X86_CP_1000M/cpu0_to_cpu16/spec2026/full
    python3 $CP_X86 --data_dir $X86_DATA_1000M --src_cpu 0 --tgt_cpu 16 --suite spec2026 --strict_loocv --out_dir $X86_CP_1000M/cpu0_to_cpu16/spec2026/top2 --input_counters $X86_CP_TOP2
    python3 $CP_X86 --data_dir $X86_DATA_1000M --src_cpu 0 --tgt_cpu 16 --suite spec2026 --strict_loocv --out_dir $X86_CP_1000M/cpu0_to_cpu16/spec2026/top4 --input_counters $X86_CP_TOP4

    echo ""; echo "=== x86 | Cross-Processor | P-Core → E-Core | DaCapo (1000M) ==="
    python3 $CP_X86 --data_dir $X86_DATA_1000M --src_cpu 0 --tgt_cpu 16 --suite dacapo --strict_loocv --out_dir $X86_CP_1000M/cpu0_to_cpu16/dacapo/full
    python3 $CP_X86 --data_dir $X86_DATA_1000M --src_cpu 0 --tgt_cpu 16 --suite dacapo --strict_loocv --out_dir $X86_CP_1000M/cpu0_to_cpu16/dacapo/top2 --input_counters $X86_CP_TOP2
    python3 $CP_X86 --data_dir $X86_DATA_1000M --src_cpu 0 --tgt_cpu 16 --suite dacapo --strict_loocv --out_dir $X86_CP_1000M/cpu0_to_cpu16/dacapo/top4 --input_counters $X86_CP_TOP4

    # ===========================================================================
    # X86 DESKTOP — CROSS-PROCESSOR — E-Core → P-Core (1000M)
    # ===========================================================================
    echo ""; echo "=== x86 | Cross-Processor | E-Core → P-Core | SPEC 2017 (1000M) ==="
    python3 $CP_X86 --data_dir $X86_DATA_1000M --src_cpu 16 --tgt_cpu 0 --suite spec2017 --strict_loocv --out_dir $X86_CP_1000M/cpu16_to_cpu0/spec2017/full
    python3 $CP_X86 --data_dir $X86_DATA_1000M --src_cpu 16 --tgt_cpu 0 --suite spec2017 --strict_loocv --out_dir $X86_CP_1000M/cpu16_to_cpu0/spec2017/top2 --input_counters $X86_CP_TOP2
    python3 $CP_X86 --data_dir $X86_DATA_1000M --src_cpu 16 --tgt_cpu 0 --suite spec2017 --strict_loocv --out_dir $X86_CP_1000M/cpu16_to_cpu0/spec2017/top4 --input_counters $X86_CP_TOP4

    echo ""; echo "=== x86 | Cross-Processor | E-Core → P-Core | SPEC 2026 (1000M) ==="
    python3 $CP_X86 --data_dir $X86_DATA_1000M --src_cpu 16 --tgt_cpu 0 --suite spec2026 --strict_loocv --out_dir $X86_CP_1000M/cpu16_to_cpu0/spec2026/full
    python3 $CP_X86 --data_dir $X86_DATA_1000M --src_cpu 16 --tgt_cpu 0 --suite spec2026 --strict_loocv --out_dir $X86_CP_1000M/cpu16_to_cpu0/spec2026/top2 --input_counters $X86_CP_TOP2
    python3 $CP_X86 --data_dir $X86_DATA_1000M --src_cpu 16 --tgt_cpu 0 --suite spec2026 --strict_loocv --out_dir $X86_CP_1000M/cpu16_to_cpu0/spec2026/top4 --input_counters $X86_CP_TOP4

    echo ""; echo "=== x86 | Cross-Processor | E-Core → P-Core | DaCapo (1000M) ==="
    python3 $CP_X86 --data_dir $X86_DATA_1000M --src_cpu 16 --tgt_cpu 0 --suite dacapo --strict_loocv --out_dir $X86_CP_1000M/cpu16_to_cpu0/dacapo/full
    python3 $CP_X86 --data_dir $X86_DATA_1000M --src_cpu 16 --tgt_cpu 0 --suite dacapo --strict_loocv --out_dir $X86_CP_1000M/cpu16_to_cpu0/dacapo/top2 --input_counters $X86_CP_TOP2
    python3 $CP_X86 --data_dir $X86_DATA_1000M --src_cpu 16 --tgt_cpu 0 --suite dacapo --strict_loocv --out_dir $X86_CP_1000M/cpu16_to_cpu0/dacapo/top4 --input_counters $X86_CP_TOP4

fi


# # ===========================================================================
# # ARM EDGE — CROSS-FREQUENCY — OOO Core (cpu4)
# # ===========================================================================
# if [[ "$TARGET" == "arm_edge" || "$TARGET" == "all" ]]; then

#     echo ""; echo "=== ARM Edge | Cross-Frequency | OOO (cpu4) | SPEC ==="
#     python3 $CF_ARM --data_dir $ARM_DATA --target_cpu 4 --suite spec --strict_loocv --out_dir $ARM_CF/cpu4/spec/full
#     # python3 $CF_ARM --data_dir $ARM_DATA --target_cpu 4 --suite spec --strict_loocv --out_dir $ARM_CF/cpu4/spec/top2 --input_counters $ARM_TOP2
#     # python3 $CF_ARM --data_dir $ARM_DATA --target_cpu 4 --suite spec --strict_loocv --out_dir $ARM_CF/cpu4/spec/top4 --input_counters $ARM_TOP4

#     echo ""; echo "=== ARM Edge | Cross-Frequency | OOO (cpu4) | DaCapo ==="
#     python3 $CF_ARM --data_dir $ARM_DATA --target_cpu 4 --suite dacapo --strict_loocv --out_dir $ARM_CF/cpu4/dacapo/full
#     # python3 $CF_ARM --data_dir $ARM_DATA --target_cpu 4 --suite dacapo --strict_loocv --out_dir $ARM_CF/cpu4/dacapo/top2 --input_counters $ARM_TOP2
#     # python3 $CF_ARM --data_dir $ARM_DATA --target_cpu 4 --suite dacapo --strict_loocv --out_dir $ARM_CF/cpu4/dacapo/top4 --input_counters $ARM_TOP4

#     # ===========================================================================
#     # ARM EDGE — CROSS-PROCESSOR — OOO → InO
#     # ===========================================================================
#     echo ""; echo "=== ARM Edge | Cross-Processor | OOO → InO | SPEC ==="
#     python3 $CP_ARM --data_dir $ARM_DATA --src_cpu 4 --tgt_cpu 1 --suite spec --strict_loocv --out_dir $ARM_CP/cpu4_to_cpu1/spec/full
#     # python3 $CP_ARM --data_dir $ARM_DATA --src_cpu 4 --tgt_cpu 1 --suite spec --strict_loocv --out_dir $ARM_CP/cpu4_to_cpu1/spec/top2 --input_counters $ARM_TOP2
#     # python3 $CP_ARM --data_dir $ARM_DATA --src_cpu 4 --tgt_cpu 1 --suite spec --strict_loocv --out_dir $ARM_CP/cpu4_to_cpu1/spec/top4 --input_counters $ARM_TOP4

#     echo ""; echo "=== ARM Edge | Cross-Processor | OOO → InO | DaCapo ==="
#     python3 $CP_ARM --data_dir $ARM_DATA --src_cpu 4 --tgt_cpu 1 --suite dacapo --strict_loocv --out_dir $ARM_CP/cpu4_to_cpu1/dacapo/full
#     # python3 $CP_ARM --data_dir $ARM_DATA --src_cpu 4 --tgt_cpu 1 --suite dacapo --strict_loocv --out_dir $ARM_CP/cpu4_to_cpu1/dacapo/top2 --input_counters $ARM_TOP2
#     # python3 $CP_ARM --data_dir $ARM_DATA --src_cpu 4 --tgt_cpu 1 --suite dacapo --strict_loocv --out_dir $ARM_CP/cpu4_to_cpu1/dacapo/top4 --input_counters $ARM_TOP4

#     # ===========================================================================
#     # ARM EDGE — CROSS-PROCESSOR — InO → OOO
#     # ===========================================================================
#     echo ""; echo "=== ARM Edge | Cross-Processor | InO → OOO | SPEC ==="
#     python3 $CP_ARM --data_dir $ARM_DATA --src_cpu 1 --tgt_cpu 4 --suite spec --strict_loocv --out_dir $ARM_CP/cpu1_to_cpu4/spec/full
#     # python3 $CP_ARM --data_dir $ARM_DATA --src_cpu 1 --tgt_cpu 4 --suite spec --strict_loocv --out_dir $ARM_CP/cpu1_to_cpu4/spec/top2 --input_counters $ARM_TOP2
#     # python3 $CP_ARM --data_dir $ARM_DATA --src_cpu 1 --tgt_cpu 4 --suite spec --strict_loocv --out_dir $ARM_CP/cpu1_to_cpu4/spec/top4 --input_counters $ARM_TOP4

#     echo ""; echo "=== ARM Edge | Cross-Processor | InO → OOO | DaCapo ==="
#     python3 $CP_ARM --data_dir $ARM_DATA --src_cpu 1 --tgt_cpu 4 --suite dacapo --strict_loocv --out_dir $ARM_CP/cpu1_to_cpu4/dacapo/full
#     # python3 $CP_ARM --data_dir $ARM_DATA --src_cpu 1 --tgt_cpu 4 --suite dacapo --strict_loocv --out_dir $ARM_CP/cpu1_to_cpu4/dacapo/top2 --input_counters $ARM_TOP2
#     # python3 $CP_ARM --data_dir $ARM_DATA --src_cpu 1 --tgt_cpu 4 --suite dacapo --strict_loocv --out_dir $ARM_CP/cpu1_to_cpu4/dacapo/top4 --input_counters $ARM_TOP4

#     # ===========================================================================
#     # ARM EDGE — CROSS-PROCESSOR — OOO → InO (100M)
#     # ===========================================================================
#     echo ""; echo "=== ARM Edge | Cross-Processor | OOO → InO | SPEC (100M) ==="
#     python3 $CP_ARM --data_dir $ARM_DATA_100M --src_cpu 4 --tgt_cpu 1 --suite spec --strict_loocv --out_dir $ARM_CP_100M/cpu4_to_cpu1/spec/full
#     # python3 $CP_ARM --data_dir $ARM_DATA_100M --src_cpu 4 --tgt_cpu 1 --suite spec --strict_loocv --out_dir $ARM_CP_100M/cpu4_to_cpu1/spec/top2 --input_counters $ARM_TOP2
#     # python3 $CP_ARM --data_dir $ARM_DATA_100M --src_cpu 4 --tgt_cpu 1 --suite spec --strict_loocv --out_dir $ARM_CP_100M/cpu4_to_cpu1/spec/top4 --input_counters $ARM_TOP4

#     echo ""; echo "=== ARM Edge | Cross-Processor | OOO → InO | DaCapo (100M) ==="
#     python3 $CP_ARM --data_dir $ARM_DATA_100M --src_cpu 4 --tgt_cpu 1 --suite dacapo --strict_loocv --out_dir $ARM_CP_100M/cpu4_to_cpu1/dacapo/full
#     # python3 $CP_ARM --data_dir $ARM_DATA_100M --src_cpu 4 --tgt_cpu 1 --suite dacapo --strict_loocv --out_dir $ARM_CP_100M/cpu4_to_cpu1/dacapo/top2 --input_counters $ARM_TOP2
#     # python3 $CP_ARM --data_dir $ARM_DATA_100M --src_cpu 4 --tgt_cpu 1 --suite dacapo --strict_loocv --out_dir $ARM_CP_100M/cpu4_to_cpu1/dacapo/top4 --input_counters $ARM_TOP4

#     # ===========================================================================
#     # ARM EDGE — CROSS-PROCESSOR — InO → OOO (100M)
#     # ===========================================================================
#     echo ""; echo "=== ARM Edge | Cross-Processor | InO → OOO | SPEC (100M) ==="
#     python3 $CP_ARM --data_dir $ARM_DATA_100M --src_cpu 1 --tgt_cpu 4 --suite spec --strict_loocv --out_dir $ARM_CP_100M/cpu1_to_cpu4/spec/full
#     # python3 $CP_ARM --data_dir $ARM_DATA_100M --src_cpu 1 --tgt_cpu 4 --suite spec --strict_loocv --out_dir $ARM_CP_100M/cpu1_to_cpu4/spec/top2 --input_counters $ARM_TOP2
#     # python3 $CP_ARM --data_dir $ARM_DATA_100M --src_cpu 1 --tgt_cpu 4 --suite spec --strict_loocv --out_dir $ARM_CP_100M/cpu1_to_cpu4/spec/top4 --input_counters $ARM_TOP4

#     echo ""; echo "=== ARM Edge | Cross-Processor | InO → OOO | DaCapo (100M) ==="
#     python3 $CP_ARM --data_dir $ARM_DATA_100M --src_cpu 1 --tgt_cpu 4 --suite dacapo --strict_loocv --out_dir $ARM_CP_100M/cpu1_to_cpu4/dacapo/full
#     # python3 $CP_ARM --data_dir $ARM_DATA_100M --src_cpu 1 --tgt_cpu 4 --suite dacapo --strict_loocv --out_dir $ARM_CP_100M/cpu1_to_cpu4/dacapo/top2 --input_counters $ARM_TOP2
#     # python3 $CP_ARM --data_dir $ARM_DATA_100M --src_cpu 1 --tgt_cpu 4 --suite dacapo --strict_loocv --out_dir $ARM_CP_100M/cpu1_to_cpu4/dacapo/top4 --input_counters $ARM_TOP4

#     # ===========================================================================
#     # ARM EDGE — CROSS-PROCESSOR — OOO → InO (1000M)
#     # ===========================================================================
#     echo ""; echo "=== ARM Edge | Cross-Processor | OOO → InO | SPEC (1000M) ==="
#     python3 $CP_ARM --data_dir $ARM_DATA_1000M --src_cpu 4 --tgt_cpu 1 --suite spec --strict_loocv --out_dir $ARM_CP_1000M/cpu4_to_cpu1/spec/full
#     # python3 $CP_ARM --data_dir $ARM_DATA_1000M --src_cpu 4 --tgt_cpu 1 --suite spec --strict_loocv --out_dir $ARM_CP_1000M/cpu4_to_cpu1/spec/top2 --input_counters $ARM_TOP2
#     # python3 $CP_ARM --data_dir $ARM_DATA_1000M --src_cpu 4 --tgt_cpu 1 --suite spec --strict_loocv --out_dir $ARM_CP_1000M/cpu4_to_cpu1/spec/top4 --input_counters $ARM_TOP4

#     echo ""; echo "=== ARM Edge | Cross-Processor | OOO → InO | DaCapo (1000M) ==="
#     python3 $CP_ARM --data_dir $ARM_DATA_1000M --src_cpu 4 --tgt_cpu 1 --suite dacapo --strict_loocv --out_dir $ARM_CP_1000M/cpu4_to_cpu1/dacapo/full
#     # python3 $CP_ARM --data_dir $ARM_DATA_1000M --src_cpu 4 --tgt_cpu 1 --suite dacapo --strict_loocv --out_dir $ARM_CP_1000M/cpu4_to_cpu1/dacapo/top2 --input_counters $ARM_TOP2
#     # python3 $CP_ARM --data_dir $ARM_DATA_1000M --src_cpu 4 --tgt_cpu 1 --suite dacapo --strict_loocv --out_dir $ARM_CP_1000M/cpu4_to_cpu1/dacapo/top4 --input_counters $ARM_TOP4

#     # ===========================================================================
#     # ARM EDGE — CROSS-PROCESSOR — InO → OOO (1000M)
#     # ===========================================================================
#     echo ""; echo "=== ARM Edge | Cross-Processor | InO → OOO | SPEC (1000M) ==="
#     python3 $CP_ARM --data_dir $ARM_DATA_1000M --src_cpu 1 --tgt_cpu 4 --suite spec --strict_loocv --out_dir $ARM_CP_1000M/cpu1_to_cpu4/spec/full
#     # python3 $CP_ARM --data_dir $ARM_DATA_1000M --src_cpu 1 --tgt_cpu 4 --suite spec --strict_loocv --out_dir $ARM_CP_1000M/cpu1_to_cpu4/spec/top2 --input_counters $ARM_TOP2
#     # python3 $CP_ARM --data_dir $ARM_DATA_1000M --src_cpu 1 --tgt_cpu 4 --suite spec --strict_loocv --out_dir $ARM_CP_1000M/cpu1_to_cpu4/spec/top4 --input_counters $ARM_TOP4

#     echo ""; echo "=== ARM Edge | Cross-Processor | InO → OOO | DaCapo (1000M) ==="
#     python3 $CP_ARM --data_dir $ARM_DATA_1000M --src_cpu 1 --tgt_cpu 4 --suite dacapo --strict_loocv --out_dir $ARM_CP_1000M/cpu1_to_cpu4/dacapo/full
#     # python3 $CP_ARM --data_dir $ARM_DATA_1000M --src_cpu 1 --tgt_cpu 4 --suite dacapo --strict_loocv --out_dir $ARM_CP_1000M/cpu1_to_cpu4/dacapo/top2 --input_counters $ARM_TOP2
#     # python3 $CP_ARM --data_dir $ARM_DATA_1000M --src_cpu 1 --tgt_cpu 4 --suite dacapo --strict_loocv --out_dir $ARM_CP_1000M/cpu1_to_cpu4/dacapo/top4 --input_counters $ARM_TOP4

# fi

echo ""; echo "All models complete. Results in: $OUT_ROOT"

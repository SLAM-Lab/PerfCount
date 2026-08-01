#!/usr/bin/env bash
# Train cross-processor translators on the DEPLOYABLE both-core counter set.
#
# The existing top4/top6 variants were selected by feature importance on the regression
# target, which lands on a frontend/branch-heavy set (branch_misses, branch_load_misses,
# branches, cpu_cycles, instructions, ref_cycles). Measured crossover AUC for that profile is
# 0.571 -- barely above chance at the only thing the scheduler needs, which is the SIGN of the
# predicted speedup ratio.
#
# This set was selected instead by AUC on the crossover itself, restricted to counters BOTH
# core types expose (Gracemont exposes 20 of the 46 the P-cores do). Measured AUC 0.734.
# It is data-memory and TLB dominated rather than frontend dominated.
#
# Note top6 is a misnomer in the existing tree: it holds 9 features (top 6 ranked plus
# baselines). This variant holds exactly the 6 named below.
set -e
cd "$(dirname "$0")"

PY=../../../.venv/bin/python3
CP_X86=cross_processor/cross_proc_x86.py
DATA=../../../processed_data_10M/x86_desktop_heterogeneous
OUT=../../../results/cross_platform/cross_proc/x86_10M
JOBS=${JOBS:-24}

COUNTERS="llc_misses branch_load_misses ref_cycles dtlb_load_misses dtlb_store_misses cache_references"
VARIANT=bothcore6

echo "=== training $VARIANT: $COUNTERS ==="
for pair in "0 16" "16 0"; do
  set -- $pair
  src=$1; tgt=$2
  dir=$([ "$src" = 0 ] && echo cpu0_to_cpu16 || echo cpu16_to_cpu0)
  for suite in spec_2017 spec_2026 dacapo_c1; do
    echo "--- $dir / $suite ---"
    $PY $CP_X86 --data_dir $DATA --out_dir $OUT/$dir/$suite/$VARIANT \
        --src_cpu $src --tgt_cpu $tgt --suite $suite \
        --input_counters $COUNTERS --jobs $JOBS --force
  done
done
echo "=== $VARIANT training complete ==="

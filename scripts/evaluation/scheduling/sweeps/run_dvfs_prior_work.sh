#!/bin/bash
# run_dvfs_prior_work.sh — DVFS prior-work comparison (no oracle bounds)
#
# Runs all practical/deployable DVFS policies on both P-cores and E-cores:
#   - Static frequency pinning (per-frequency baselines)
#   - Reactive 1-step lookback
#   - Industry OS governors: Performance, Ondemand, Conservative, Schedutil/PELT, Intel HWP
#   - Academic online: EWMA, UCB1, Random
#
# Oracle policies (Greedy_Oracle, MPC_Oracle, Global_Oracle) are excluded —
# they require future timing data and serve as upper bounds, not prior work.
#
# Usage:
#   ./run_dvfs_prior_work.sh
#   OUTPUT_DIR=/my/out ./run_dvfs_prior_work.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec "$SCRIPT_DIR/run.sh" --policies \
    Static_P_1.0GHz Static_P_2.0GHz Static_P_3.0GHz Static_P_4.0GHz \
    Static_E_1.0GHz Static_E_2.0GHz Static_E_3.0GHz Static_E_4.0GHz \
    Performance_Gov_P \
    Ondemand_P      Conservative_P      Schedutil_PELT_P      Intel_HWP_P \
    Ondemand_E      Conservative_E      Schedutil_PELT_E \
    EWMA_P          UCB1_P              Random_P \
    EWMA_E          UCB1_E              Random_E \
    "$@"

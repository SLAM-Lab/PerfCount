"""Measured physical parameters of the platforms the simulator replays.

Everything here is a property of hardware, not of a scheduling policy. Each value is either a
measurement (with the benchmark that produced it named) or an explicit estimate (labelled as
such). Keeping them separate from the orchestration in `main.py` means a number can be traced
to its source without reading the simulator, and re-measuring one platform cannot disturb the
policy code.

Three cost families are modelled:

  DVFS transition   changing frequency on one core. Measured latency and energy per
                    (from, to) frequency pair, per core type.
  Core migration    moving between the P and E clusters. A fixed context-switch latency plus
                    a frequency-pair-dependent energy.
  Cache warmup      the transient slowdown after a migration, as a decaying multiplicative
                    penalty. ESTIMATED, not measured -- see WARMUP below.

Environment overrides (all default to 1.0, leaving the measured model untouched) scale the
REALISED cost so the conclusions can be tested against a harsher machine:

  WARMUP_A_SCALE     multiplies both warmup amplitudes (depth of the slowdown)
  WARMUP_TAU_SCALE   multiplies both warmup time constants (how long it persists)
  MIG_LAT_SCALE      multiplies the P<->E context-switch latency

These differ from --warmup_decision_scale, which biases only what a policy *believes* a
migration costs, not what it is charged.
"""
import math
import os

# ---------------------------------------------------------------------------
# Sensitivity scaling (read once, at import)
# ---------------------------------------------------------------------------
WARMUP_A_SCALE = float(os.environ.get('WARMUP_A_SCALE', 1.0))
WARMUP_TAU_SCALE = float(os.environ.get('WARMUP_TAU_SCALE', 1.0))
MIG_LAT_SCALE = float(os.environ.get('MIG_LAT_SCALE', 1.0))
DVFS_COST_SCALE = float(os.environ.get('DVFS_COST_SCALE', 1.0))  # scales DVFS freq-transition lat+energy; 0 = free switching

# ---------------------------------------------------------------------------
# x86: Intel Core i7-13700K  (P = Golden Cove / cpu0, E = Gracemont / cpu16)
# ---------------------------------------------------------------------------

# --- Cache warmup after a cross-cluster migration --------------------------
# ESTIMATED, not measured. The incremental cost of a migration beyond the context switch is
# below our measurement floor: isolating it means resolving a few nanojoules and microseconds
# against RAPL's 50 ms update period and a 10M-instruction sample.
#
# Reasoning behind the values, for 10M-instruction chunks on this part. L3 is shared across the
# clusters, so only L1d/L2, the branch predictors and the hardware prefetchers go cold:
#     L2 cold-start          (1-4M instr warmup)     ~10-15% peak slowdown
#     BTB/BHT miss spike     (200K-1M instr)         ~3-8%  (gcc/perlbench worst)
#     prefetcher retraining  (50-200K accesses)      ~3-5%  (memory-bound workloads)
# P->E is the larger penalty: the E cluster shares 2 MB of L2 across four cores (~0.5 MB
# effective) against 1.25 MB on a P core, and Golden Cove's prefetch/BTB units warm faster on
# the way back. tau is in CHUNK units, and warmup completes in roughly 3-4M instructions, so
# 0.3-0.4 of a chunk.
#
# Setting both amplitudes to 0.0 reproduces the no-penalty baseline exactly, which is how the
# sensitivity sweeps bracket this model.
WARMUP_A_PtoE = 0.20 * WARMUP_A_SCALE     # peak slowdown entering the E cluster
WARMUP_TAU_PtoE = 0.4 * WARMUP_TAU_SCALE  # decay constant, in chunks
WARMUP_A_EtoP = 0.12 * WARMUP_A_SCALE     # peak slowdown entering the P cluster
WARMUP_TAU_EtoP = 0.3 * WARMUP_TAU_SCALE
# Chunks penalised after a migration. Three time constants captures essentially all of the
# decaying ramp, so the floor of 10 only binds when TAU_SCALE stretches it further.
WARMUP_K = max(10, int(math.ceil(3.0 * max(WARMUP_TAU_PtoE, WARMUP_TAU_EtoP))))

# --- P<->E migration --------------------------------------------------------
# Context switch: mean 4.47 us, symmetric within 0.1 us
# (ctx_switch_bench, 10 repetitions x 5000 migrations).
MIG_LAT_S = 4.47e-6 * MIG_LAT_SCALE

# Migration energy (J) keyed by the (P frequency, E frequency) pair in effect at the migration:
# MIG_LAT_S x the median package power (RAPL power/energy-cores) measured while ctx_switch_bench
# migrates continuously at that pair
# (power_collection/ctx_switch/freq_sweep_power/ -> analyze_migration_energy.py).
#
# This replaced an idle-subtraction approach (duty-cycled benchmark against a same-core control)
# that tried to recover the ~1-10 nJ of a single migration as a delta against RAPL's ~50 ms,
# ~10-50 mW noise floor. That produced sign-flipping, order-of-magnitude-unstable estimates.
# Measuring active power directly (O(1-25 W)) avoids the noise floor, at the cost of folding in
# some background power that is not specific to the migration.
MIG_NRG_J = {
    (1.0, 1.0): 5.35e-06,  (1.0, 2.0): 8.92e-06,  (1.0, 3.0): 2.139e-05, (1.0, 4.0): 8.909e-05,
    (2.0, 1.0): 8.93e-06,  (2.0, 2.0): 1.25e-05,  (2.0, 3.0): 2.589e-05, (2.0, 4.0): 9.637e-05,
    (3.0, 1.0): 1.875e-05, (3.0, 2.0): 2.054e-05, (3.0, 3.0): 2.852e-05, (3.0, 4.0): 0.00010253,
    (4.0, 1.0): 4.012e-05, (4.0, 2.0): 5.893e-05, (4.0, 3.0): 5.974e-05, (4.0, 4.0): 0.0001061,
}

# --- DVFS frequency transitions ---------------------------------------------
# Latency (us) from simple_latency.c + analyze_dvfs.py, 100 repetitions per pair, median across
# repetitions. The median is used because downscale and E-core transitions have a long right
# tail, where the core occasionally steps through an intermediate P-state.
#
# Pairs that measured a median of 0.00 completed inside a single compute kernel iteration, below
# the rolling-median detection resolution. Those are floored to one iteration period at the
# target frequency (P-core 3 GHz = 2.68 us, E-core 4 GHz = 3.01 us, E-core 3 GHz = 4.02 us) and
# marked below.
P_LAT_US = {
    (1.0, 2.0): 4.02,  (1.0, 3.0): 2.68,  (1.0, 4.0): 2.01,
    (2.0, 1.0): 7.93,  (2.0, 3.0): 2.68,  (2.0, 4.0): 2.01,
    (3.0, 1.0): 8.04,  (3.0, 2.0): 2.68,  (3.0, 4.0): 2.01,
    (4.0, 1.0): 8.04,  (4.0, 2.0): 2.22,  (4.0, 3.0): 2.68,   # floored (median 0.00)
}
E_LAT_US = {
    (1.0, 2.0): 6.03,  (1.0, 3.0): 8.02,  (1.0, 4.0): 6.04,
    (2.0, 1.0): 12.04, (2.0, 3.0): 4.02,  (2.0, 4.0): 3.01,
    (3.0, 1.0): 12.04, (3.0, 2.0): 4.01,  (3.0, 4.0): 3.01,   # floored (median 0.00)
    (4.0, 1.0): 12.06, (4.0, 2.0): 3.69,  (4.0, 3.0): 4.02,   # floored (median 0.00)
}

# Transition energy (J) = 95th-percentile package power across the transition x its latency.
P_NRG_J = {
    (1.0, 2.0): 2.1e-05,  (1.0, 3.0): 1.6e-05,  (1.0, 4.0): 1.3e-05,
    (2.0, 1.0): 5.8e-05,  (2.0, 3.0): 2.1e-05,  (2.0, 4.0): 1.8e-05,
    (3.0, 1.0): 8.4e-05,  (3.0, 2.0): 5.7e-05,  (3.0, 4.0): 2.6e-05,
    (4.0, 1.0): 0.000117, (4.0, 2.0): 6.6e-05,  (4.0, 3.0): 4.7e-05,
}
E_NRG_J = {
    (1.0, 2.0): 2.3e-05,  (1.0, 3.0): 1.6e-05,  (1.0, 4.0): 1.5e-05,
    (2.0, 1.0): 5.8e-05,  (2.0, 3.0): 2.2e-05,  (2.0, 4.0): 2.0e-05,
    (3.0, 1.0): 7.1e-05,  (3.0, 2.0): 4.4e-05,  (3.0, 4.0): 2.9e-05,
    (4.0, 1.0): 0.0002,   (4.0, 2.0): 0.000132, (4.0, 3.0): 9.7e-05,
}

# Fallbacks for frequency pairs absent from the tables above.
DVFS_LAT_S = 5.0e-6
DVFS_NRG_J = 2e-5

# DVFS_COST_SCALE scales every frequency-transition latency+energy (default 1.0 = measured model).
# Setting it to 0 makes frequency switching free, which isolates how much of a greedy policy's gap
# to the Viterbi oracle is unplanned transition cost.
if DVFS_COST_SCALE != 1.0:
    P_LAT_US = {k: v * DVFS_COST_SCALE for k, v in P_LAT_US.items()}
    E_LAT_US = {k: v * DVFS_COST_SCALE for k, v in E_LAT_US.items()}
    P_NRG_J = {k: v * DVFS_COST_SCALE for k, v in P_NRG_J.items()}
    E_NRG_J = {k: v * DVFS_COST_SCALE for k, v in E_NRG_J.items()}
    DVFS_LAT_S *= DVFS_COST_SCALE
    DVFS_NRG_J *= DVFS_COST_SCALE

# ---------------------------------------------------------------------------
# Arm big.LITTLE: Qualcomm RB5 / Snapdragon 865
#   L = Little (Silver, Cortex-A55), B = Big (Gold, Cortex-A76)
# ---------------------------------------------------------------------------
# PLACEHOLDERS. None of these are measured on the RB5; the warmup values are copied from the
# x86 estimates and the transition tables are empty, so every pair falls back to ARM_DVFS_*.
# The x86 microbenchmarks port directly, but the board exposes no per-cluster energy counter,
# so the energy terms need the whole-board instrument.
ARM_WARMUP_A_BtoL = 0.20
ARM_WARMUP_TAU_BtoL = 0.4
ARM_WARMUP_A_LtoB = 0.12
ARM_WARMUP_TAU_LtoB = 0.3
ARM_WARMUP_K = 10

ARM_MIG_LAT_S = 5.0e-6
ARM_MIG_NRG_J = {(1.0, 1.0): 5.0e-06}
ARM_DVFS_LAT_S = 5.0e-6
ARM_DVFS_NRG_J = 2e-5

ARM_L_LAT_US = {}
ARM_L_NRG_J = {}
ARM_B_LAT_US = {}
ARM_B_NRG_J = {}

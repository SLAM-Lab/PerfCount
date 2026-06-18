# Scheduling Simulation Framework

End-to-end pipeline for evaluating DVFS and heterogeneous scheduling policies
on x86 desktop (Alder Lake) P-core / E-core traces.

## Hardware Setup

| Role | Core | CPU ID | Frequencies |
|------|------|--------|-------------|
| P-core | Golden Cove | `cpu0` | 1.0, 2.0, 3.0, 4.0 GHz |
| E-core | Gracemont | `cpu16` | 1.0, 2.0, 3.0, 4.0 GHz |

## Pipeline Overview

```
Stage 0: Data Collection (on target machine)
    │
    ▼
Stage 1: Process Raw Data ──► aligned HW-counter traces
    │
    ▼
Stage 2: Merge Power Data ──► aligned traces + RAPL power columns
    │
    ▼
Stage 3: Generate Speedup Matrix ──► per-config speedup traces
    │
    ├──────────────────────────────────┐
    ▼                                  ▼
Stage 4a/b: Train Prediction      Stage 6: Simulator
    Models (cross-freq,               (oracle/heuristic
     cross-proc)                       policies only)
    │
    ▼
Stage 5a/b: Precompute Model
    Predictions
    │
    ▼
Stage 6: Simulator (full run with model policies)
```

---

## Stage 0 — Data Collection

Collection scripts generate shell scripts that drive `perf record` on the
target machine.  Two collection runs are needed per (benchmark, core, freq):

- **Run 1–N** (HW counters): groups of perf events collected in separate passes
  (`my_run=1..N`), with instruction-block granularity (10M instructions/block).
- **Run 100** (power): 3 events (instructions, cpu_cycles, ref_cycles) +
  simultaneous RAPL power sampling via `spec_power_wrapper.sh`.

### Scripts

| Script | Purpose |
|--------|---------|
| `scripts/data_collection/x86_desktop_heterogeneous/spec_2017_collection.py` | Generate SPEC 2017 collection shell script |
| `scripts/data_collection/x86_desktop_heterogeneous/spec_2026_collection.py` | Generate SPEC 2026 collection shell script |
| `scripts/data_collection/x86_desktop_heterogeneous/spec_power_wrapper.sh` | Co-records perf + RAPL power with sync timestamp |

### Output (per benchmark × core × freq × phase)

```
raw_data_10M/x86_desktop_heterogeneous/
    cpu_<cpu>_<freq>GHz_<bench>_10000000_<run>_<phase>.out   # HW counter runs

power_data_10M/x86_desktop_heterogeneous/
    cpu_<cpu>_<freq>GHz_<bench>_10000000_100_<phase>.out     # perf binary (run=100)
    cpu_<cpu>_<freq>GHz_<bench>_10000000_100_<phase>_power.csv
    cpu_<cpu>_<freq>GHz_<bench>_10000000_100_<phase>_sync.txt
    idle_<cpu>_<freq>GHz_power.csv                           # idle power baseline
```

---

## Stage 1 — Process Raw Data

Parses raw perf `.out` files, merges split instruction blocks, repairs dropped
samples, and aligns multi-run counter groups into one CSV per
(benchmark, freq, core, phase).

```bash
python scripts/data_processing/x86_desktop_heterogeneous/process_raw_data.py \
  --raw_dir  raw_data_10M/x86_desktop_heterogeneous \
  --out_dir  processed_data_10M/x86_desktop_heterogeneous \
  --jobs $(nproc)
```

### Output

```
processed_data_10M/x86_desktop_heterogeneous/
    aligned_<bench>_<freq>GHz_cpu<N>_phase<P>.csv
    # Columns: sample_index, 22 HW counters (instructions, cpu_cycles, ref_cycles, ...)
```

---

## Stage 2 — Build Power-Annotated Traces

Aligns RAPL power samples (from run=100) with the perf instruction blocks
and adds `power_watts_total_block` (plus 3 other power columns) to each
aligned trace.

**This stage is required for per-sample power mode** (`--power_mode per_sample`
in Stage 6).  `generate_speedup_matrix.py` (Stage 3) looks for
`power_watts_total_block` in the input traces and propagates it into
`Power_<config>` columns in the speedup CSVs.  Without it, the simulator
falls back to a static power lookup table.

### How it works

The power collection run (run=100) records 3 perf events (instructions,
cpu_cycles, ref_cycles) simultaneously with RAPL power sampling, producing
three files per (bench, core, freq, phase):

- `.out` — perf binary with instruction-block boundaries
- `_power.csv` — RAPL power samples with timestamps
- `_sync.txt` — monotonic clock reference for aligning the two

The script parses the instruction blocks from the `.out` file, aligns them
with the RAPL power trace using the sync timestamp, computes idle-relative
and total (idle + active) power per block and as a rolling average, and
writes the 4 power columns.

The original aligned traces in `processed_data_10M/` are left untouched —
they are used as-is for model training (Stage 4), and changing their
dimensions would break inference.  The merged output goes to a separate
`processed_data_10M_power/` directory.

```bash
python scripts/data_processing/x86_desktop_heterogeneous/build_power_aligned_traces.py \
  --power_dir   power_data_10M/x86_desktop_heterogeneous \
  --aligned_dir processed_data_10M/x86_desktop_heterogeneous \
  --output_dir  processed_data_10M_power/x86_desktop_heterogeneous \
  --workers 8
```

### Output

```
processed_data_10M_power/x86_desktop_heterogeneous/
    aligned_<bench>_<freq>GHz_cpu<N>_phase<P>.csv
    # All 22+ HW counters from Stage 1 + 4 power columns:
    #   power_watts_block, power_watts_rolling_10,
    #   power_watts_total_block, power_watts_total_rolling_10
```

### Per-sample power data flow

```
Stage 2: aligned trace (power_watts_total_block column)
    ──► Stage 3: generate_speedup_matrix.py ──► speedups CSV (Power_<config> columns)
        ──► Stage 6: simulator data_loader (--power_mode per_sample)
            ──► per-chunk energy = time × power
```

---

## Stage 3 — Generate Speedup Matrix

Reads all aligned traces (needs `ref_cycles`; uses `power_watts_total_block`
if present), pivots per-config execution time, computes pairwise speedup
ratios and measured power per block, and writes the granular phase-level
traces consumed by the simulator.

```bash
python scripts/evaluation/scheduling/utils/generate_speedup_matrix.py \
  --input_dir  processed_data_10M_power/x86_desktop_heterogeneous \
  --output_dir results/scheduling/speedup_full \
  --workers 8
```

### Output

```
results/scheduling/speedup_full/
    granular_phase_traces/
        speedups_<config>_<bench>_phase<P>.csv    # e.g. speedups_P_3.0GHz_spec_500.perlbench_r_phase0.csv
    condensed_average_speedups_summary.csv
```

---

## Stage 4 — Train Prediction Models

CatBoost LOOCV models that predict execution time at a target configuration
from PMU counters measured at a source configuration. Two axes, trained
per suite (spec2017, spec2026, dacapo).

### 4a. Cross-Frequency (same core type, different freq)

Run once per (core, suite) combination:

```bash
# P-core models (cpu0)
for suite in spec2017 spec2026 dacapo; do
  python scripts/evaluation/cross_platform_prediction/cross_frequency/cross_freq_x86.py \
    --data_dir  processed_data_10M/x86_desktop_heterogeneous \
    --out_dir   results/cross_platform/cross_freq/x86_10M/cpu0/${suite}/full \
    --target_cpu 0 --suite ${suite} --save_predictions
done

# E-core models (cpu16)
for suite in spec2017 spec2026 dacapo; do
  python scripts/evaluation/cross_platform_prediction/cross_frequency/cross_freq_x86.py \
    --data_dir  processed_data_10M/x86_desktop_heterogeneous \
    --out_dir   results/cross_platform/cross_freq/x86_10M/cpu16/${suite}/full \
    --target_cpu 16 --suite ${suite} --save_predictions
done
```

### 4b. Cross-Processor (different core type)

Run once per (direction, suite) combination:

```bash
for suite in spec2017 spec2026 dacapo; do
  # P → E
  python scripts/evaluation/cross_platform_prediction/cross_processor/cross_proc_x86.py \
    --data_dir  processed_data_10M/x86_desktop_heterogeneous \
    --out_dir   results/cross_platform/cross_proc/x86_10M/cpu0_to_cpu16/${suite}/full \
    --src_cpu 0 --tgt_cpu 16 --suite ${suite} --save_predictions

  # E → P
  python scripts/evaluation/cross_platform_prediction/cross_processor/cross_proc_x86.py \
    --data_dir  processed_data_10M/x86_desktop_heterogeneous \
    --out_dir   results/cross_platform/cross_proc/x86_10M/cpu16_to_cpu0/${suite}/full \
    --src_cpu 16 --tgt_cpu 0 --suite ${suite} --save_predictions
done
```

### Feature toggles (shared across all model scripts)

| Flag | Default | Description |
|------|---------|-------------|
| `--use_mpki / --no_mpki` | on | Misses-per-kilo-instruction features |
| `--use_miss_rates / --no_miss_rates` | on | Cache/TLB miss rate features |
| `--use_stall_rates / --no_stall_rates` | on | Frontend/backend stall rate features |
| `--use_bottleneck_class / --no_bottleneck_class` | on | Categorical bottleneck feature |
| `--rolling_window` | 5 | Rolling-mean smoothing window |
| `--strict_loocv / --no_strict_loocv` | on | Hold out entire workload (not just fold) |
| `--equal_weight` | off | Equal-weight workloads during training |
| `--exclude_features` | none | Comma-separated features to drop |

---

## Stage 5 — Precompute Model Predictions

Offline batch inference: applies trained models to PMU traces and writes
predicted speedup CSVs in the format expected by the simulator's data loader.

### 5a. Cross-Frequency Precompute

The `--suites` flag controls which suites to process (defaults to all three).
Pass `--model_base_dir` pointing to the top-level cross-freq results directory;
the script resolves `<cpu>/<suite>/full/` subdirs automatically.

```bash
# P-core predictions (all suites)
python scripts/evaluation/scheduling/cross_freq_precompute.py \
  --model_base_dir results/cross_platform/cross_freq/x86_10M \
  --pmu_dir        processed_data_10M/x86_desktop_heterogeneous \
  --oracle_dir     results/scheduling/speedup_full/granular_phase_traces \
  --out_dir        results/scheduling/cross_freq_predictions \
  --core_type P

# E-core predictions (all suites)
python scripts/evaluation/scheduling/cross_freq_precompute.py \
  --model_base_dir results/cross_platform/cross_freq/x86_10M \
  --pmu_dir        processed_data_10M/x86_desktop_heterogeneous \
  --oracle_dir     results/scheduling/speedup_full/granular_phase_traces \
  --out_dir        results/scheduling/cross_freq_predictions \
  --core_type E
```

To run a single suite only: `--suites spec2026`

### 5b. Cross-Processor Precompute

Same `--suites` flag; resolves `cpu0_to_cpu16/<suite>/full/` and
`cpu16_to_cpu0/<suite>/full/` automatically.

```bash
python scripts/evaluation/scheduling/cross_proc_precompute.py \
  --model_dir  results/cross_platform/cross_proc/x86_10M \
  --pmu_dir    processed_data_10M/x86_desktop_heterogeneous \
  --oracle_dir results/scheduling/speedup_full/granular_phase_traces \
  --out_dir    results/scheduling/cross_proc_predictions
```

### Output

```
results/scheduling/cross_freq_predictions/
    speedups_from_P_<ghz>GHz/speedups_P_<ghz>GHz_<bench>_phase<P>.csv
    speedups_from_E_<ghz>GHz/speedups_E_<ghz>GHz_<bench>_phase<P>.csv

results/scheduling/cross_proc_predictions/
    speedups_from_P_<ghz>GHz/<bench>_phase<P>.csv
    speedups_from_E_<ghz>GHz/<bench>_phase<P>.csv
```

---

## Stage 6 — Run the Simulator

Evaluates all scheduling policies (oracle bounds, reactive heuristics,
model-based, combined DVFS+migration) across every workload-phase and
generates comparison plots, taxonomy charts, and summary CSVs.

```bash
python scripts/evaluation/scheduling/src/main.py \
  --input_dir             results/scheduling/speedup_full/granular_phase_traces \
  --output_dir            results/scheduling/model_out_v10 \
  --cross_freq_p_pred_dir results/scheduling/cross_freq_predictions \
  --cross_freq_e_pred_dir results/scheduling/cross_freq_predictions \
  --cross_proc_pred_dir   results/scheduling/cross_proc_predictions \
  --viterbi_cache_dir     results/scheduling/viterbi_cache \
  --apply_warmup
```

### Flags

| Flag | Required | Description |
|------|----------|-------------|
| `--input_dir` | yes | Granular speedup traces (Stage 3 output) |
| `--output_dir` | yes | Where to write results |
| `--power_mode` | no | `per_sample` (default) or `baseline` |
| `--cross_freq_p_pred_dir` | no | P-core model predictions (Stage 5a). Enables `Model_Greedy_P`, `Model_Greedy_Oracle_P`, `Model_Global_P` |
| `--cross_freq_e_pred_dir` | no | E-core model predictions (Stage 5a). Enables `Model_Greedy_E`, `Model_Greedy_Oracle_E`, `Model_Global_E` |
| `--cross_proc_pred_dir` | no | Cross-proc predictions (Stage 5b). Enables `Model_IsoFreq_*`, `IsoFreq_Model_Oracle_*` |
| `--viterbi_cache_dir` | no | Cache/reuse global-oracle DP traces across runs |
| `--apply_warmup` | no | Apply cache-warmup time penalty after P↔E migrations |

### Output

```
results/scheduling/model_out_v10/
    all_phases_summary.csv              # raw per-(workload, phase, metric, policy) results
    diagnostics.csv                     # per-policy action statistics
    csv/
        suite_avg_all_policies.csv      # suite-level means (unnormalized)
        suite_avg_{P_DVFS,E_DVFS,IsoFreq,Hetero}.csv
        per_workload_avg.csv
        per_workload_{P_DVFS,E_DVFS,IsoFreq,Hetero}.csv
        taxonomy_*                      # per-category taxonomy breakdowns
    bar_suite/                          # suite-averaged bar plots (normalized to oracle)
    bar_wl/                             # per-workload bar plots
    taxonomy/
        suite_level/                    # taxonomy charts (reactive / forecast / perfect)
        {spec_2017,spec_2026,...}/
            per_workload/               # per-workload taxonomy charts
```

---

## Quick Reference — Full Pipeline Commands

```bash
# 1. Process raw HW-counter data
python3 scripts/data_processing/x86_desktop_heterogeneous/process_raw_data.py \
  --raw_dir raw_data_10M/x86_desktop_heterogeneous \
  --out_dir processed_data_10M/x86_desktop_heterogeneous \
  --jobs $(nproc)

# 2. Merge RAPL power into aligned traces
python3 scripts/data_processing/x86_desktop_heterogeneous/build_power_aligned_traces.py \
  --power_dir   power_data_10M/x86_desktop_heterogeneous \
  --aligned_dir processed_data_10M/x86_desktop_heterogeneous \
  --output_dir  processed_data_10M_power/x86_desktop_heterogeneous \
  --workers 8

# 3. Generate speedup matrix (from power-annotated traces)
python3 scripts/evaluation/scheduling/utils/generate_speedup_matrix.py \
  --input_dir  processed_data_10M_power/x86_desktop_heterogeneous \
  --output_dir results/scheduling/speedup_full \
  --workers 8

# 4a. Train cross-freq models (P-core and E-core, all suites)
for suite in spec2017 spec2026 dacapo; do
  python3 scripts/evaluation/cross_platform_prediction/cross_frequency/cross_freq_x86.py \
    --data_dir processed_data_10M/x86_desktop_heterogeneous \
    --out_dir  results/cross_platform/cross_freq/x86_10M/cpu0/${suite}/full \
    --target_cpu 0 --suite ${suite} --save_predictions

  python3 scripts/evaluation/cross_platform_prediction/cross_frequency/cross_freq_x86.py \
    --data_dir processed_data_10M/x86_desktop_heterogeneous \
    --out_dir  results/cross_platform/cross_freq/x86_10M/cpu16/${suite}/full \
    --target_cpu 16 --suite ${suite} --save_predictions
done

# 4b. Train cross-proc models (both directions, all suites)
for suite in spec2017 spec2026 dacapo; do
  python3 scripts/evaluation/cross_platform_prediction/cross_processor/cross_proc_x86.py \
    --data_dir processed_data_10M/x86_desktop_heterogeneous \
    --out_dir  results/cross_platform/cross_proc/x86_10M/cpu0_to_cpu16/${suite}/full \
    --src_cpu 0 --tgt_cpu 16 --suite ${suite} --save_predictions

  python3 scripts/evaluation/cross_platform_prediction/cross_processor/cross_proc_x86.py \
    --data_dir processed_data_10M/x86_desktop_heterogeneous \
    --out_dir  results/cross_platform/cross_proc/x86_10M/cpu16_to_cpu0/${suite}/full \
    --src_cpu 16 --tgt_cpu 0 --suite ${suite} --save_predictions
done

# 5a. Precompute cross-freq predictions (all suites)
python3 scripts/evaluation/scheduling/cross_freq_precompute.py \
  --model_base_dir results/cross_platform/cross_freq/x86_10M \
  --pmu_dir        processed_data_10M/x86_desktop_heterogeneous \
  --oracle_dir     results/scheduling/speedup_full/granular_phase_traces \
  --out_dir        results/scheduling/cross_freq_predictions --core_type P

python3 scripts/evaluation/scheduling/cross_freq_precompute.py \
  --model_base_dir results/cross_platform/cross_freq/x86_10M \
  --pmu_dir        processed_data_10M/x86_desktop_heterogeneous \
  --oracle_dir     results/scheduling/speedup_full/granular_phase_traces \
  --out_dir        results/scheduling/cross_freq_predictions --core_type E

# 5b. Precompute cross-proc predictions (all suites)
python3 scripts/evaluation/scheduling/cross_proc_precompute.py \
  --model_dir  results/cross_platform/cross_proc/x86_10M \
  --pmu_dir    processed_data_10M/x86_desktop_heterogeneous \
  --oracle_dir results/scheduling/speedup_full/granular_phase_traces \
  --out_dir    results/scheduling/cross_proc_predictions

# 6. Run simulator
python3 scripts/evaluation/scheduling/src/main.py \
  --input_dir             results/scheduling/speedup_full/granular_phase_traces \
  --output_dir            results/scheduling/model_out_v10 \
  --cross_freq_p_pred_dir results/scheduling/cross_freq_predictions \
  --cross_freq_e_pred_dir results/scheduling/cross_freq_predictions \
  --cross_proc_pred_dir   results/scheduling/cross_proc_predictions \
  --viterbi_cache_dir     results/scheduling/viterbi_cache \
  --apply_warmup
```

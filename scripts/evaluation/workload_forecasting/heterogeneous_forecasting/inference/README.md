# Cross-config workload forecasting

Predict a benchmark's future `ref_cycles` at a **target** config
`C = (core_t, freq_t)` given history observed at a **source** config
`S = (core_s, freq_s)`, via **translate-then-forecast**:

1. take S's observed window,
2. translate `ref_cycles` into C's space with the cross-platform model
   (cross-processor when cores differ, cross-frequency when only freq differs;
   identity when `S == C`), **copying the other top-4 counters unchanged**,
3. run C's own pre-trained per-config forecaster,
4. score against C's aligned ground-truth future (MAPE).

No new training — it reuses the trained per-config forecasters and the
cross-platform translation models already on disk.

## What we compare (per target config C)
For each target C we compute three things, all scored as MAPE against **C's
ground-truth future**:

- **`oracle`** — C's forecaster on **C's own** history = *homogeneous forecasting*,
  the best achievable ("if you were actually running on C"). This is the floor.
- **`translated`** — C's forecaster on a **foreign** source S's history, unified to
  C = *forecasting a setup you're not on*.
- **`naive`** — same as translated but without unification (shows translation's value).

The quantity of interest is the **gap `translated − oracle`**: the price of
predicting a config you're not currently running on. The experiment is: given a
heterogeneous history at S, unify it to each of the N configs and predict all N
futures — the matrices below are that source→target sweep.

## Location
```
scripts/evaluation/workload_forecasting/cross_config/
├── predict_cross_config.py   # harness: one benchmark, all 64 config pairs
├── run_cross_config.sh       # driver: sweeps benchmarks × models × H × T
├── analyze_cross_config.py   # turns the CSV into source→target matrices
└── README.md
```

## Quick start (two steps: run the matrix, then analyze it)
```bash
cd /home/meb4744/PerfCount/scripts/evaluation/workload_forecasting/cross_config

# 1. run the FULL config matrix (all 64 src->tgt pairs) for every benchmark
./run_cross_config.sh                        # DT, horizon 1, timesteps 5

# 2. view it as cross-frequency + cross-processor matrices
./analyze_cross_config.py                    # or: python analyze_cross_config.py
```
Step 1 writes `results/forecasting/cross_config/cross_config_10M.csv` (overwritten
each run). Step 2 reads it and prints the category summary + per-core cross-frequency
matrices + cross-processor (P→E, E→P) matrices, each with the oracle floor per target.

The full config matrix is swept **per invocation** — `run_cross_config.sh` only loops
the extra dimensions (benchmark, model, horizon, timesteps).

### Sweep more configs (env vars)
```bash
MODELS="dt mlp lstm transformer" HORIZONS="1 5 10" TIMESTEPS="5 10" ./run_cross_config.sh
BENCHES="dacapo_avrora spec_505.mcf_r" ./run_cross_config.sh   # subset of benchmarks
```
The driver runs single-threaded (pinned env vars set inside). A full sweep is large
(benchmarks × models × H × T × 56 config pairs); start with the defaults.

## Run a single case directly (the harness)
Useful for spot-checks. `--benchmark` is the bench "rest" name (no `aligned_`/config).
```bash
cd /home/meb4744/PerfCount/scripts/evaluation/workload_forecasting
export PYTHONPATH=$PWD
../../../.venv/bin/python3 cross_config/predict_cross_config.py \
    --benchmark dacapo_avrora --model dt --horizon 1 --timesteps 5

# restrict to one source/target config (cpu:freq):
... --benchmark spec_505.mcf_r --src 0:4.0 --tgt 16:4.0
```
Config codes: cpu `0` = P-core, `16` = E-core; freq in GHz (`1.0 2.0 3.0 4.0`).
With no `--src/--tgt` it evaluates all 8×8 = 64 config pairs (56 cross + 8 identity).

## Output columns
`src_cpu, src_freq, tgt_cpu, tgt_freq, bench, model, horizon, timesteps, method, mape`

`method` ∈:
| method | meaning |
|---|---|
| `translated` | proposed: `ref_cycles` translated S→C, other counters copied |
| `oracle` | C's forecaster on C's own history (upper bound) |
| `naive` | C's forecaster on S's untranslated history |
| `persistence` | carry the last (translated) `ref_cycles` forward |

## What it reuses (inputs it expects on disk)
- Forecasters: `results/forecasting/models_10M/x86_desktop_heterogeneous_cpu{0,16}_top4/{freq}GHz/horizon_{h}/timesteps_{t}/{workload}_{model}.{pkl,keras}`
- ref_cycles translators: `results/cross_platform/cross_{proc,freq}/x86_10M/...`
- Aligned traces: `processed_data_10M/x86_desktop_heterogeneous/**/aligned_{bench}_{freq}GHz_cpu{cpu}_phase*.csv`

If a forecaster or translator is missing for a pair, that pair is skipped (no crash).

## Known result / caveat
Translating `ref_cycles` alone **fully recovers cross-frequency** (translated ≈ oracle),
and helps cross-core, but for the **cross-core** axis the copied `cpu_cycles` (which the
forecaster leans on heavily) caps accuracy — worst for E→P (predicting the fast P-core
from slow E-core history). Extending translation to `cpu_cycles`
(`cross_platform_prediction/cross_processor/cross_proc_counter_translation.py` already
supports any target counter) is the next step.

# Phase-Aware Workload Forecasting (x86_desktop)

Single home for the phase-aware forecasting study on `x86_desktop_heterogeneous` (dataset-
parameterized via `--dataset`; target config cpu0 @ 4.0 GHz by default). Everything runs on the
**config-invariant top4b** counter set `{ref_cycles, cpu_cycles, branches, instructions}` — only
`ref_cycles` (the target) and `cpu_cycles` (core-dependent) need translation across configs.

**One command runs the whole study:** [`run_forecasting.sh`](run_forecasting.sh).

## The story arc

- **A. Homogeneous forecasting + counter reduction** — full counters → top4 → **top4b**
  (`branches` replaces the config-dependent `branch_misses`; DT accuracy improved).
- **B. Plain (non-phase) heterogeneous** — cross-config *inference* (translate a foreign observed
  history, forecast the target) and *training* injection. Lives in `../heterogeneous_forecasting/inference/`
  (`predict_cross_config.py`); translators under `results/cross_platform/{cross_freq,cross_proc}/`.
- **C. Phase-aware forecasting (this directory)**
  - **C1/C2 homogeneous** — per-phase models + selection gate; big win on *multi-phase* workloads
    (MLP −33% vs global); explicit next-phase *prediction* is a null result at 10M granularity.
    **Delta** targets (predict `x[i+1]-x[i]`) make persistence the floor → DT+delta+gate **beats
    persistence**, especially at phase transitions.
  - **C3 heterogeneous-history TRAINING** — inject foreign donors into training, naive vs
    translated. Naive is catastrophic; translation recovers ~85% of the gap; transition-win holds
    to ~30% injection.
  - **C4 heterogeneous INFERENCE** — cold prediction of a config you only observe through a foreign
    one. Honest baseline = **translated-persistence** (homogeneous persistence is unavailable cold).
    The phase-aware forecaster **beats translated-persistence** in every cell, both directions,
    wider at transitions.

See [`RESULTS.md`](RESULTS.md) for the numbers.

## Scripts

| script | role |
|---|---|
| [`run_forecasting.sh`](run_forecasting.sh) | **master** — runs Stages 1-4 end to end (all models/benches/probs/directions) |
| `predict_phase_forecast.py` | core harness: per-phase + global + gated forecasters, delta, transition-weighted metric (C1/C2/C3) |
| `predict_het_inference.py` | het-inference harness: train on target C, evaluate on translated/naive foreign inputs vs translated-persistence (C4) |
| `run_all_models.sh` | homogeneous sweep, all 4 models, parallel; `DELTA=1` for delta, `SAVE=1` to persist |
| `run_het_phase.sh` | het-training sweep (`MODE`, `HET_PROB`); regimes homogeneous/naive/translated |
| `run_het_inference.sh` | het-inference sweep (`SOURCES` = both directions) |
| `analyze_phase_forecast.py` | per-method + per-suite table |
| `analyze_multiphase.py` | multi- vs single-phase split (where phase-awareness pays off) |
| `analyze_het_phase.py` | het-training regime comparison + translation recovery % |
| `analyze_het_inference.py` | het-inference: forecaster vs translated-persistence, @all and @transition |

Model handling is automatic: LSTM runs `--stateless`; NN models skip the (expensive) k-fold gate
and use `NN_EPOCHS` epochs; DT keeps the gate.

## Running

```bash
cd scripts/evaluation/workload_forecasting/phase_forecasting

# Full study (all ~68 benches, all models, prob sweep, both directions) — run from tmux:
./run_forecasting.sh

# Fast DT-only end-to-end pass:
MODELS=dt ./run_forecasting.sh

# Quick smoke on two benches:
BENCHES="spec_505.mcf_r dacapo_h2" MODELS=dt PROBS=0.3 ./run_forecasting.sh
```

Env knobs: `MODELS`, `PROBS`, `DIRECTIONS`, `BENCHES`, `PAR`, `NN_EPOCHS`, `SAVE`, `FREQ`, `CPU`,
`DATASET`. The full matrix with NN models is large (hours-to-days); DT stages finish first.

**Parallelism**: jobs are single-threaded (`OMP/MKL/TF` threads pinned to 1), so `PAR` = concurrent
jobs. Default `PAR=80` (this is a 160-core box). Watch RAM on all-NN stages (~1.5–2 GB/job →
~120–160 GB at `PAR=80`, of ~190 GB free).

**Het-injection sweep**: `PROBS="0.2 0.4 0.6 0.8 1.0"` — from 20% up to a fully heterogeneous
training set (prob=1.0, every training window is a foreign donor).

## Dependencies (shared infra, unchanged)
- Counter translators: `results/cross_platform/{cross_freq,cross_proc}/x86_10M/` (used for C3/C4).
  Cross-freq root is **per-cpu** (`.../cross_freq/x86_10M/cpu0`); cross-proc uses the tree +
  `counter_translation/` subdir.
- Traces: `processed_data_10M/<dataset>/…/aligned_<bench>_<freq>GHz_cpu<cpu>_phase*.csv`.
- Forecasting/classify pipeline: `../src/` (get_raw_data, TimeSeries, Predictor, classify, evaluate).
- Translation helpers reused from `../heterogeneous_forecasting/inference/predict_cross_config.py`.

## Outputs (`results/forecasting/phase_forecasting/`)
- `phase_forecast_10M_<model>_gmm[_delta].csv` — homogeneous (rows: bench, method,
  phase∈{all,0..k,transition,steady}, mape, n, cov, regime).
- `het_phase_<dir>_<model>_delta_p<prob>.csv` — het-training (regime = homogeneous/naive/translated).
- `het_infer_<model>[_delta].csv` — het-inference (regime = oracle/translated/naive).
- `models/<bench>/<tag>/` — saved ensembles (`global.*`, `phase{p}.*`, `metadata.json`) when `SAVE=1`.

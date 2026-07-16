# Forecasting Under Heterogeneity (x86 desktop)

Regenerates every result behind Chapter 5's **Forecasting Under Heterogeneity** section, and the
figures that report them.

A forecaster trained at a fixed operating point breaks when a scheduler moves the workload, because
counter magnitudes are operating-point dependent: frequency scaling compresses cycle counts
nonlinearly and the two core types differ in pipeline behavior. The model then sees counters drawn
from one configuration paired with a target drawn from another. Translating the counters into the
target configuration repairs it. This directory measures that, on both sides of the model.

**One command runs the whole study:**
[`run_forecasting_under_heterogeneity.sh`](run_forecasting_under_heterogeneity.sh).

This study does **not** touch the scheduling simulator. The sim consumes its own prediction dumps
(`../phase_forecasting/run_dump_dvfs.sh` → `scheduling/utils/cap_predictions.py` →
`scheduling/run_x86_sweep.sh`); nothing here feeds it. See [Relation to the simulator](#relation-to-the-simulator).

## The two studies

- **TRAIN — training-time heterogeneity.** Replace each row of a training trace, with probability
  $p \in \{0, 0.2, \dots, 1.0\}$, by the same-position row of a donor trace of the same benchmark at
  a different operating point. The donor configuration is drawn independently per row, so the
  perturbed history mixes several sources at once — the general case a migrating scheduler
  accumulates. `p=0` recovers the homogeneous baseline; `p=1` is full replacement. Naive swap vs
  translated swap, both cores, two donor pools (`cross_freq` = same core other frequency,
  `cross_proc` = other core any frequency).

- **INFER — inference-time heterogeneity.** Translate-then-forecast over all 56 ordered
  source→target pairs of the 8 configurations (2 cores × 4 frequencies): take the window observed at
  source `S`, unify it into target `C`'s counter space, feed it to `C`'s existing forecaster. Two
  arms — translating `ref_cycles` alone, then `cpu_cycles` additionally across cores. Scored against
  an **oracle** (`C` forecast from its own history, the homogeneous floor) and a **naive** baseline
  (untranslated foreign window).

## Flow

```
  results/cross_platform/                            <- PREREQUISITE, built elsewhere
    cross_freq/x86_10M/cpu{0,16}/{suite}/top4/...        ref_cycles across frequency
    cross_proc/x86_10M/cpu{s}_to_cpu{t}/...              ref_cycles across cores
    cross_proc/x86_10M/counter_translation/...           cpu_cycles across cores
          |
          |  ../../cross_platform_prediction/run_x86.sh
          |  ../../cross_platform_prediction/run_counter_translation.sh
          v
  +---------------------------- this script ----------------------------+
  |                                                                     |
  |  --only train            --only infer                               |
  |  training/               inference/                                 |
  |    run_het_cross_freq_     run_cross_config.sh  x2 arms             |
  |      sweeps.sh               VARIANT=modelcmp                       |
  |    run_het_cross_proc_       TRANSLATE="ref_cycles"                 |
  |      sweeps.sh               TRANSLATE="ref_cycles cpu_cycles"      |
  |         |                        |                                  |
  |         v                        v                                  |
  |  logs_10M/x86_desktop_    results/forecasting/cross_config/        |
  |    heterogeneous_cpu{0,16}  cross_config_10M_modelcmp.csv           |
  |    _het_{cross_freq,        cross_config_10M_modelcmp_refcpu.csv    |
  |     cross_proc}_                                                    |
  |    {naive,translated}/                                              |
  |         |                        |                                  |
  |         +----------+-------------+                                  |
  |                    v                                                |
  |              --only figs                                            |
  |    plotting_scripts/forecasting/                                    |
  |      heterogeneous_training/plot_translation_results.py             |
  |                            -> results_cross_{frequency,             |
  |                                  processor}[_ecore]*.pdf            |
  |      heterogeneous_inference/plot_inference_summary.py              |
  |                            -> results_inference_summary.pdf         |
  |                    |                                                |
  |                    v  (only the three Chapter 5 references)         |
  |                 figures/                                            |
  +---------------------------------------------------------------------+
```

## What it needs

Everything is read from `results/cross_platform/`. **Nothing here trains a translator** — build those
first with `../../cross_platform_prediction/run_x86.sh` and `run_counter_translation.sh`.

| Input | Used by | Consequence if absent |
|---|---|---|
| `cross_freq/x86_10M/cpu{0,16}/{spec_2017,spec_2026,dacapo_c1}/top4/` | both | that suite is **silently dropped** from the cross-frequency average |
| `cross_proc/x86_10M/cpu{s}_to_cpu{t}/` | both | cross-processor pairs skipped |
| `cross_proc/x86_10M/counter_translation/cpu{s}_to_cpu{t}/` | INFER, `_refcpu` arm | `cpu_cycles` is silently **copied** instead of translated |
| aligned traces `processed_data_10M/x86_desktop_heterogeneous/` | both | benchmark skipped |
| per-config forecasters `results/forecasting/models_10M/..._top4/` | INFER | benchmark skipped |

Run `--check` first. It verifies each tree, prints per-suite model coverage, and flags results older
than the newest translator.

```bash
./run_forecasting_under_heterogeneity.sh --check
```

### Why `--check` exists

**Every failure here is silent.** A missing translator makes the harness skip the pair, so a whole
suite disappears from an average rather than raising. This has already bitten twice:

- `create_dataset.py:_bench_suite_dir_cross_freq` once pointed DaCapo at `dacapo_c1_pruned`, which
  covers only 17 of 22 benchmarks. The five it lacks (`cassandra`, `h2o`, `kafka`, `tradebeans`,
  `tradesoap`) got no translator, were skipped by the dump, and the simulator then **substituted
  oracle times** for them — reading as a near-perfect model result rather than an error. Fixed; the
  docstring records it.
- `cross_config_10M_modelcmp_refcpu.csv` was generated 2026-07-13, a day *before* the cross-frequency
  translators were retrained. SPEC CPU2026 had zero translated cross-frequency rows in it, so the
  reported "naive 37.2% vs translated 15.4%" compared naive over three suites against translated over
  two. Matched on the same pairs it is 39.7% vs 15.4%.

Neither produced an error message. `--check` is the only thing standing between you and the third one.

## Usage

```bash
./run_forecasting_under_heterogeneity.sh --check      # prerequisites only, run nothing
./run_forecasting_under_heterogeneity.sh              # train + infer + figs
./run_forecasting_under_heterogeneity.sh --only infer # one stage
./run_forecasting_under_heterogeneity.sh --only figs  # replot from existing logs/CSVs
DRY=1 ./run_forecasting_under_heterogeneity.sh        # print commands, execute nothing

MODELS="dt" ./run_forecasting_under_heterogeneity.sh --only infer   # quick DT-only check
```

| Env | Default | Meaning |
|---|---|---|
| `MODELS` | `dt mlp lstm transformer` | models for the INFER sweep |
| `HORIZONS` | `1` | forecast horizon |
| `TIMESTEPS` | `5` | history window |
| `MAX_WORKERS` | *(launcher default)* | forwarded to the TRAIN launchers |
| `DRY` | `0` | `1` prints commands without running |

TRAIN is the long pole: four sweeps (naive/translated × two cores) per donor pool, each over
benchmarks × models × horizons × `p`. INFER is benchmarks × models × 56 pairs.

## Verifying the output

```bash
inference/analyze_cross_config.py        # source->target matrices + oracle floor per target
```

Check that **all three suites appear** in the cross-frequency average. If SPEC CPU2026 is missing,
its translators were not found and the number is computed over two suites.

## Known gaps

- `run_counter_translation.sh` only writes `cpu16_to_cpu0`. The `cpu0_to_cpu16` tree exists but that
  script will not rebuild it, so a full regeneration leaves one direction older than the other.
- SPEC CPU2026 cross-**processor** models are being retrained (1 GHz results under investigation).
  The INFER stage will regenerate those cells; hold it or disregard them until the retrain lands.

## Relation to the simulator

Chapter 5's scheduling policies do **not** reuse the per-configuration forecasters this study
evaluates. The simulator translates a single observed source's whole trace and fits its own
forecaster on it by walk-forward
(`../phase_forecasting/dump_dvfs_forecast.py`). That places the scheduler's forecaster at the
`p=1.0` translated condition of the TRAIN study, but with a single source rather than the per-row
donor mixture, so the TRAIN `p=1.0` numbers **bound** the scheduler's forecast quality from above
rather than estimating it. Regenerating this study does not invalidate the sim, and regenerating the
sim does not require rerunning this.

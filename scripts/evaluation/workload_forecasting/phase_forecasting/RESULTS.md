# Phase-Level Workload Forecasting — Results

Homogeneous, single-config phase-aware forecasting (after Alcorta et al., SAMOS'21 /
TODAES'22). Config: `x86_desktop_heterogeneous`, cpu0 (P-core) @ 4.0 GHz, GMM phases
(`--phase_count 6`), target `ref_cycles`, timesteps 5, horizon 1. Balanced 24-benchmark
set: 8 DaCapo / 8 SPEC2017 / 8 SPEC2026.

All numbers are MAPE (%), lower is better. "MAPE is scale-invariant, and 10M-instruction
bins make ref_cycles CPI-proportional, so this is comparable to CPI-based forecasting."

## Pipeline / code
- Harness: [`predict_phase_forecast.py`](predict_phase_forecast.py) — trains models fresh each
  run (does NOT reuse the cross-config `models_10M` forecasters).
- Launcher: [`run_all_models.sh`](run_all_models.sh) — all 4 models, parallel across benches.
  Env: `MODELS`, `CLASSIFIER`, `PHASE_COUNT`, `NN_EPOCHS`, `DELTA` (1=residual targets),
  `SAVE` (1=persist ensembles), `PAR`, `BENCHES`.
- Analyzers: [`analyze_phase_forecast.py`](analyze_phase_forecast.py) (per-method + per-suite),
  [`analyze_multiphase.py`](analyze_multiphase.py) (multi- vs single-phase split).

## Methods compared
| method | what it does |
|---|---|
| `global` | one forecaster on all training windows (non-phase-aware baseline) |
| `per_phase` | route each test window to a model trained on its CURRENT phase |
| `per_phase_pred` | route to the LEARNED next-phase prediction (Alcorta-style) |
| `per_phase_oracle_next` | route to the TRUE next phase (upper bound of phase prediction) |
| `per_phase_gated` | per_phase, but keep a phase's model only if it beats global on k-fold within-train validation |
| `persistence` | carry last value forward (the wall at h=1) |

`--delta` mode forecasts the residual over persistence (predict x[i+1]−x[i], reconstruct
x[i]+Δ). Rows are also split into `@transition` (phase changes to next window) vs `@steady`.

## Key findings

### 1. Phase-awareness helps on MULTI-PHASE workloads (15/24 benches, eff. phases ≥ 2)
MLP, multi-phase benches: `global` 15.71 → `per_phase` **10.55** (−5.16, −33% rel). On
single-phase benches per_phase is neutral-to-harmful (+1.13) — no structure to exploit.
Aggregate 24-bench means dilute this; the benefit concentrates in the multi-phase subset.

### 2. Explicit next-phase PREDICTION does not help at 10M granularity
Transition rate is only ~5–16%, so the current phase is a near-perfect proxy for the next.
Even a perfect oracle-next beats current-phase routing by only ~0.2–0.3 MAPE; the learned
next-phase predictor (72% acc) is slightly worse than current-phase routing. Model-invariant
(holds for DT and MLP). Value comes from the per-phase MODELS, not the phase predictor.

### 3. Delta targets + gating BEATS persistence (DT), esp. at transitions
DT, 24 benches:
| | global | per_phase | per_phase_gated | persistence |
|---|---|---|---|---|
| absolute @all | 11.47 | 9.68 | 8.77 | **6.38** |
| **delta @all** | 6.34 | 7.25 | **6.15** | 6.38 |
| delta @all (multi-phase) | 7.72 | 8.73 | **7.43** | 7.94 |
| **delta @transition (multi)** | 13.10 | 14.04 | **13.00** | 15.93 |
| delta @steady | 5.73 | 6.75 | 5.52 | **5.33** |

`per_phase_gated` + `--delta` beats persistence overall (6.15 vs 6.38), on multi-phase
(7.43 vs 7.94), and clearly at the transition windows that matter for proactive scheduling
(13.0 vs 15.9). Persistence still wins at steady state — fine, you don't reschedule when
nothing changes. The gate is ESSENTIAL in delta mode (ungated per_phase hurts).

### 4. Model notes
- MLP per_phase (absolute) is more stable than DT (worst regression +3.2 vs +19.4) — nearly
  matches DT's gated result without a gate.
- MLP+delta at 30 epochs did NOT beat persistence (global 8.74 vs 6.38): the small residual
  target undertrains a lightly-trained MLP, whereas a DT leaf degrades gracefully to ≈0
  (=persistence). Re-running MLP+delta at 100 epochs to test convergence (see below).

## Output locations
- Per-run CSVs: `results/forecasting/phase_forecasting/phase_forecast_10M_<model>_<classifier>[_delta].csv`
  (rows: bench, method, phase∈{all,0..k,transition,steady}, mape, n, cov).
- Saved ensembles (`SAVE=1`): `results/forecasting/phase_forecasting/models/<bench>/<model>_<classifier>[_delta]/`
  — `global.*`, `phase{p}.*`, `next_phase_clf.joblib`, `metadata.json` (config + per-method MAPE).

## Reproduce
```bash
cd scripts/evaluation/workload_forecasting/phase_forecasting
# absolute, all 4 models, save ensembles:
SAVE=1 ./run_all_models.sh
# delta (residual-over-persistence), MLP only, 100 epochs, save:
MODELS=mlp DELTA=1 NN_EPOCHS=100 SAVE=1 ./run_all_models.sh
# analyze:
./analyze_multiphase.py            # multi- vs single-phase split
```

## Open directions
- MLP/LSTM+delta at full epoch budget (convergence vs persistence).
- Step 2: phase-transition-imminent feature to sharpen transition-window accuracy.
- Heterogeneous bridge: per-phase forecasting on translated cross-config histories.

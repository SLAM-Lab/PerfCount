# Scheduling / DVFS Policy Reference

Every policy evaluated by `src/main.py`, grouped by problem and explained. The
simulator scores each policy on **EDP** and **ED²P** per workload-phase and
normalizes to an oracle bound.

There are three problem families:
- **Single-core DVFS** — pick the frequency on one core (`_P` = P-core/Golden Cove,
  `_E` = E-core/Gracemont), 1.0–4.0 GHz.
- **Cross-processor iso-frequency migration** (`IsoFreq_*`) — fix the frequency,
  decide **which core** (P vs E) to run on each chunk.
- **Full heterogeneous** (`*_Hetero`) — jointly decide core **and** frequency.

---

## The three axes every policy is built from

**1. Temporal axis — *what data the policy sees* at decision time `i`:**

| temporal | sees | realizable? |
|---|---|---|
| **reactive** | the **previous** chunk (`i-1`) | yes (deployable) |
| **oracle** | the **current/future** true chunk (`i…i+k`) | no (upper bound) |
| **forecast** | a **prediction** of chunk `i` from history `< i` | yes (this thesis) |

**2. Prediction-source axis — *what estimates the cost*:**

| source | mechanism |
|---|---|
| **proxy signal** | a hardware activity signal (governors/heuristics) |
| **model** | CatBoost cross-config translation of PMU counters → target-config time |
| **forecast** | walk-forward DT forecast of the target-config time (`dump_dvfs_forecast.py`) |
| **true** | the measured ground-truth time (oracle only) |

**3. Decision axis — *how it acts on the estimate*:**

| decision | behavior |
|---|---|
| **greedy** | pick the best config for the (single) visible chunk |
| **k-lookahead** | average the next `k+1` chunks, then pick |
| **MPC** | plan a transition-cost-aware path over a horizon `W` |
| **global** (Viterbi) | full-trace DP — the unconstrained optimum |
| **dampened** (`DampN`) | greedy, but blend predictions toward their rolling-`N` mean when volatile (anti-chatter hysteresis) |

**The evaluation ladder** most results are read against (best→worst realizable):
`oracle (true time)` → `perfect-counter (model @ true chunk i)` → `FORECAST (predicted i)` → `reactive-model (model @ i-1)` → `governors` → `static`.
Everything is normalized so the relevant oracle = 1.0 (lower = better).

**Naming shorthand:** `_Future_` = the same heuristic given the current chunk's
*true* proxy (perfect-future heuristic bound). `_Oracle` = uses true/current data.
`Model_` = uses cross-config model predictions. `Forecast` = uses walk-forward
forecasts. `DampN` = rolling-window-`N` dampening. `kN`/`WN` = lookahead/horizon.

---

## 1. Single-core DVFS

### Static baselines
| policy | meaning |
|---|---|
| `Static_{P,E}_{1-4}.0GHz` | run the whole trace pinned at one fixed frequency. Reference points; the best static is the "no-DVFS" baseline. |

### Deployable governors / heuristics (reactive, proxy-signal)
All decide from **past** proxy samples only.
| policy | meaning |
|---|---|
| `Ondemand_{P,E}` | Linux ondemand — ramp to max on high utilization, step down when idle. |
| `Conservative_{P,E}` | Linux conservative — like ondemand but steps frequency gradually. |
| `Schedutil_PELT_{P,E}` | schedutil driven by a PELT-style utilization estimate. |
| `Intel_HWP_{P,E}` | Intel Hardware P-states (speed-shift) response model. |
| `EWMA_{P,E}` | exponentially-weighted-moving-average utilization governor. |
| `UCB1_{P,E}` | UCB1 bandit over frequencies (explores/exploits). |
| `Performance_Gov_{P,E}` | always max frequency (Linux performance governor). |
| `*_Future_{P,E}` | **perfect-future** variant of each governor: same algorithm fed the *current* chunk's true proxy instead of the past one. An upper bound on what the heuristic could do with a perfect signal. |

### Oracle bounds (true data)
| policy | meaning |
|---|---|
| `Reactive_Oracle_{P,E}` | repeat the **previous** chunk's *truly-optimal* frequency (perfect knowledge, one chunk stale). The honest reactive ceiling. |
| `Greedy_Oracle_{P,E}` | pick each chunk's truly-optimal frequency (perfect per-chunk knowledge). Per-chunk optimum. |
| `Global_Oracle_{P,E}` | full-trace Viterbi DP with transition costs — the unconstrained DVFS optimum. |

### Model-based (CatBoost cross-frequency predictions)
Fed via `--cross_freq_{p,e}_pred_dir`.
| policy | temporal | meaning |
|---|---|---|
| `Model_Greedy_{P,E}` | reactive (`i-1`) | **the reactive-model baseline** — translate the *actual* last-chunk PMU to every target frequency, pick the best. Deployable. |
| `Model_Greedy_Oracle_{P,E}` | oracle (`i`) | **perfect-counter** — same model, but on the *current* chunk's true PMU. Bounds "how good is the model with perfect timing." |
| `Model_Greedy_Oracle_k{1,2,5}_{P,E}` | oracle (`i…i+k`) | perfect-counter with k-step lookahead (average next k+1 chunks). |
| `Model_Global_{P,E}` | — | Viterbi over the model predictions (model-based global optimum). |

### Forecast (this thesis — walk-forward DT)
Fed via `--cross_freq_{p,e}_forecast_dir`.
| policy | temporal | meaning |
|---|---|---|
| `Model_Forecast_{P,E}` | forecast (`i`) | **the DVFS workload-forecasting policy** — greedy on the causal walk-forward forecast of each target frequency's time. The deployable "predict the future" policy; compare against `Model_Greedy` (reactive-model) and `Greedy_Oracle`. |

---

## 2. Cross-processor iso-frequency migration (`IsoFreq_*`)

Frequency is fixed at `{1,2,3,4}.0GHz`; the policy chooses **P vs E** each chunk.
Migration costs a context switch (4.47 µs) + a post-migration cache-warmup penalty
(`--apply_warmup`).

### Oracle bounds (true data)
| policy | meaning |
|---|---|
| `IsoFreq_Oracle_{f}` | truly-optimal core each chunk at freq `f` (cost-aware). **The normalizer** for the migration ladder. |
| `IsoFreq_Reactive_Oracle_{f}` | repeat the previous chunk's truly-optimal core. Reactive ceiling. |
| `IsoFreq_Oracle_Heuristic_{f}` | EAS-style heuristic given the current chunk's *true* timing/energy (perfect-future heuristic). |

### Model-based migration (cross-proc CatBoost predictions)
Fed via `--cross_proc_pred_dir`.
| policy | temporal / decision | meaning |
|---|---|---|
| `Model_IsoFreq_{f}` | reactive (`i-1`) | **reactive-model migration baseline** — translate last-chunk PMU cross-core, pick the better core. |
| `Model_IsoFreq_Damp{5,10}_{f}` | reactive + dampened | reactive-model with rolling-window hysteresis — **the strongest deployable baseline** (avoids migrating on transient spikes under migration cost). |
| `IsoFreq_Model_Oracle_{f}` | oracle (`i`) | perfect-counter migration (model at the true current chunk). |
| `IsoFreq_Model_Oracle_k{1,2,5}_{f}` | oracle (`i…i+k`) | perfect-counter with k-step lookahead. |
| `IsoFreq_Model_MPC_Oracle_W{5,10}_{f}` | oracle + MPC | transition-cost-aware planning over horizon `W` on the true future — the cost-aware upper bound. |

### Forecast migration (this thesis)
Fed via `--cross_proc_forecast_dir`.
| policy | temporal / decision | meaning |
|---|---|---|
| `Model_IsoFreq_Forecast_{f}` | forecast (`i`) | **migration forecasting policy** — greedy on the causal walk-forward cross-proc forecast. |
| `Model_IsoFreq_Forecast_Damp{5,10}_{f}` | forecast + dampened | dampened forecast — combines the forecast's trajectory signal with anti-chatter hysteresis. The best-performing forecast variant for migration. |

---

## 3. Full heterogeneous (core **and** frequency) — `*_Hetero`

Fed by the merged cross-freq + cross-proc predictions.

### Deployable heuristics
| policy | meaning |
|---|---|
| `EAS_Hetero` | Energy-Aware Scheduling — migrate toward the more energy-efficient core by a utilization heuristic. |
| `EAS_With_DVFS` | EAS jointly choosing core and frequency. |
| `Thread_Director` | Intel Thread Director-style class-based placement. |
| `Threshold_Migration` | migrate when a utilization/proxy threshold is crossed. |
| `Micro_EAS` | fine-grained per-chunk EAS variant. |
| `UCB1_Hetero` | UCB1 bandit over the full (core × freq) action set. |

### Model-based heterogeneous
| policy | temporal / decision | meaning |
|---|---|---|
| `Model_Reactive_Hetero` | reactive (`i-1`) | reactive-model over the full config set. |
| `Model_Reactive_Damp{5,10}_Hetero` | reactive + dampened | dampened reactive-model (anti-chatter). |
| `Model_Greedy_Oracle_Hetero` | oracle (`i`) | perfect-counter over the full config set. |
| `Model_Greedy_Oracle_k{1,2,5}_Hetero` | oracle (`i…i+k`) | perfect-counter with lookahead. |
| `Model_MPC_Oracle_W{5,10}_Hetero` | oracle + MPC | cost-aware planning over the full config set. |

### Oracle bounds
| policy | meaning |
|---|---|
| `Greedy_Oracle_Hetero` | truly-optimal (core, freq) each chunk (greedy). |
| `Proactive_Hetero_Oracle` | full-trace Viterbi over (core, freq) with transition costs — the unconstrained heterogeneous optimum. |
| `EAS_Oracle_Hetero`, `Thread_Director_Oracle` | perfect-future variants of the corresponding heuristics. |

---

## 4. Combined DVFS + migration (explicit two-action)
| policy | meaning |
|---|---|
| `Reactive_Combined_W{1,5,10}` | reactive combined DVFS+migration controller over a lookback window `W`. |
| `MPC_Oracle_Combined_W{1,5,10}` | MPC combined controller with the true future over horizon `W` (planning upper bound). |

---

## How to read a comparison

For a given problem (e.g. iso-freq migration at 4 GHz), normalize each policy to
its oracle (`IsoFreq_Oracle_4.0GHz = 1.0`) and line up:

```
reactive-model (Model_IsoFreq)        deployable floor (translate last chunk)
  dampened reactive (…_Damp10)        strongest deployable baseline
FORECAST (Model_IsoFreq_Forecast)     this thesis (predict chunk i)
  dampened forecast (…_Forecast_Damp) best forecast variant
perfect-counter (IsoFreq_Model_Oracle)  model with perfect timing
perfect-future k5 / MPC-oracle        cost-aware upper bounds
oracle (IsoFreq_Oracle)               = 1.0
```

The **reactive → forecast → perfect-counter** span is the *forecasting-addressable*
headroom; the **perfect-counter → oracle** span is model-accuracy / decision-rule
headroom that better *forecasting* cannot close. Governors and static baselines sit
well above the model policies and mark the deployable state of the art.

> **Note on model type:** all `Forecast` policies currently use a **DT** walk-forward
> forecaster; all `Model_` (non-forecast) policies use CatBoost cross-config
> translation. See `dump_dvfs_forecast.py` (`--model`) to swap the forecaster.

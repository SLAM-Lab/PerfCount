#!/usr/bin/env bash
# Run the heterogeneous scheduler with the FAIR ("self_forecast") diagonal and check whether it
# holds up, or whether it thrashes the way the previous-sample ("prev") diagonal did.
#
# Background: the incumbent config's "stay" score is the diagonal of the prediction tensor.
#   oracle (shipped)  = the TRUE next sample (an idealization no scheduler has; favours staying)
#   prev              = the previous sample (BROKEN: backward-looking stay vs forward-looking moves
#                       -> 4-10x migration thrash; the "significant" deployable gain was the gate
#                       merely damping that thrash, not real forecasting skill)
#   self_forecast     = the incumbent's OWN causal forecast (forward-looking, same basis as the
#                       move scores, fully deployable) -- the fair diagonal this script tests
#
# HARD RULE, enforced by the report below: look at n_transitions BEFORE any EDP number. If
# self_forecast's transition counts sit near the oracle baseline, the diagonal is consistent and
# the gate-vs-reactive numbers are trustworthy. If they blow up like prev did, the number is an
# artifact and must not go in the chapter.
#
#   ./run_selffc.sh            run the general arm (missing only), then report
#   ARMS="general loocv" ./run_selffc.sh   also run loocv
#   FORCE=1 ./run_selffc.sh    re-run even if the summary exists
set -e
cd "$(dirname "$0")"

PY=../../../.venv/bin/python3
RES=../../../results/scheduling
PMU=../../../processed_data_10M/x86_desktop_heterogeneous
HP=$RES/Hetero_precompute
DP=$RES/DVFS_precompute
TRACES=$HP/speedup_full_v2_repaired/granular_phase_traces
VITERBI=$HP/viterbi_cache_hetero
SF=$(cd "$RES" && pwd)/self_forecast_10M          # absolute: the loader reads it directly
export SIM_WORKERS=${SIM_WORKERS:-20}
export OMP_NUM_THREADS=1
export DIAGONAL_MODE=self_forecast
export SELF_FORECAST_DIR=$SF
ARMS=${ARMS:-general}

# Bench set + exclusions, identical to run_chapter5.sh so the numbers are comparable arm-to-arm.
BENCHES=$( { ls $TRACES | grep -oE 'spec_[0-9]+\.[A-Za-z0-9]+_r' | sort -u
             find $PMU/dacapo_c1 -name 'aligned_dacapo_*_cpu0_phase*.csv' -printf '%f\n' 2>/dev/null \
               | grep -oE 'dacapo_[A-Za-z0-9]+' | sort -u; } | sort -u | paste -sd' ')
EXCL=$( comm -23 <(ls $TRACES | grep -oE 'spec_[0-9]+\.[A-Za-z0-9]+_r|dacapo_[A-Za-z0-9]+' | sort -u) \
                 <(printf '%s\n' $BENCHES | sort -u) | paste -sd, )
NB=$(echo $BENCHES | wc -w)

# Fail loud if the self-forecast dump is incomplete: a missing file falls back to the oracle
# diagonal for that config, silently mixing diagonals and contaminating the comparison.
NPH=$(ls $TRACES/speedups_P_1.0GHz_*.csv 2>/dev/null | wc -l)
for c in P_1.0GHz P_2.0GHz P_3.0GHz P_4.0GHz E_1.0GHz E_2.0GHz E_3.0GHz E_4.0GHz; do
  n=$(ls "$SF/$c"/*.csv 2>/dev/null | wc -l)
  [ "$n" -ge "$NPH" ] || { echo "INCOMPLETE self-forecast dump: $c has $n/$NPH files ($SF/$c)" >&2
      echo "regenerate with: MODE=self_forecast CORES='0 16' FREQS='1.0 2.0 3.0 4.0' OUT_DIR=$SF \\" >&2
      echo "  bash ../workload_forecasting/phase_forecasting/run_dump_dvfs.sh" >&2; exit 1; }
done
echo "self-forecast dump OK ($NPH phases x 8 configs) ; $NB benches ; SIM_WORKERS=$SIM_WORKERS"

sim () {  # $1 = arm name (general|loocv); writes hetero/${1}_selffc
  local out="${1}_selffc"
  if [ -z "$FORCE" ] && [ -f "$RES/hetero/$out/all_phases_summary.csv" ]; then
    echo "  hetero/$out present, skipping (FORCE=1 to redo)"; return; fi
  local cp_tr cp_fc
  case "$1" in
    general) cp_tr=$HP/cross_proc_translate_gentemporal_10M; cp_fc=$HP/cross_proc_forecast_gentemporal_10M ;;
    loocv)   cp_tr=$HP/cross_proc_translate_10M;             cp_fc=$HP/cross_proc_forecast_10M ;;
    *) echo "unknown arm $1" >&2; exit 1 ;;
  esac
  echo "=== hetero/$out (DIAGONAL_MODE=self_forecast) ==="
  FAST_HETERO=1 $PY src/main.py --input_dir $TRACES --output_dir $RES/hetero/$out \
    --power_mode per_sample --decision_power_mode static --warmup_in_decision --apply_warmup \
    --strict_predictions --viterbi_cache_dir $VITERBI ${EXCL:+--exclude_workloads $EXCL} \
    --cross_freq_p_pred_dir $DP/cross_freq_translate_gentemporal_10M --cross_freq_e_pred_dir $DP/cross_freq_translate_gentemporal_10M \
    --cross_freq_p_forecast_dir $DP/cross_freq_forecast_gentemporal_10M --cross_freq_e_forecast_dir $DP/cross_freq_forecast_gentemporal_10M \
    --cross_proc_pred_dir $cp_tr --cross_proc_forecast_dir $cp_fc
}

for a in $ARMS; do sim "$a"; done

echo; echo "=== GUARDRAIL: transition counts first (oracle diagonal vs self_forecast) ==="
RES=$RES $PY - "$ARMS" <<'PY'
import os, sys, re, pandas as pd
RES = os.environ['RES']; arms = sys.argv[1].split()
EXC = {'spec_772.marian_r', 'spec_706.stockfish_r'}
def suite(w):
    if w.startswith('dacapo'): return 'DaCapo'
    m = re.match(r'spec_(\d+)', w); return 'SPEC2017' if m and int(m.group(1)) < 700 else 'SPEC2026'
NORM='Proactive_Hetero_Oracle'; REACT='Model_Reactive_Hetero'; GATE='Model_Forecast_ReactiveGated_Hetero'
def diag(dirn):
    f = os.path.join(RES, 'hetero', dirn, 'diagnostics.csv')
    return pd.read_csv(f) if os.path.exists(f) else None
def summ(dirn):
    f = os.path.join(RES, 'hetero', dirn, 'all_phases_summary.csv')
    return pd.read_csv(f) if os.path.exists(f) else None
for a in arms:
    do, dn = diag(a), diag(a + '_selffc')
    if do is None or dn is None:
        print(f"  {a}: need diagnostics for both hetero/{a} and hetero/{a}_selffc"); continue
    print(f"  -- {a}: mean n_transitions (oracle -> self_forecast); >~2x = thrash, reject --")
    for pol in (REACT, GATE):
        o = do[(do.Metric=='EDP') & (do.Policy==pol)].n_transitions.mean()
        s = dn[(dn.Metric=='EDP') & (dn.Policy==pol)].n_transitions.mean()
        flag = '' if (o and s/o < 2) else '   <-- THRASH, do not trust EDP'
        print(f"       {pol.replace('Model_',''):34s} {o:8.0f} -> {s:8.0f}  ({s/o:.2f}x){flag}")

print("\n=== EDP ladder (only meaningful if transitions above are NOT thrashing) ===")
for a in arms:
    e = summ(a + '_selffc')
    if e is None: continue
    e = e[(~e.Workload.isin(EXC)) & (e.Metric=='EDP')]; e['S']=e.Workload.map(suite)
    print(f"  -- {a}_selffc, gate vs reactive over oracle --")
    for s in ['SPEC2017','SPEC2026']:
        p = e[e.S==s].pivot_table(index=['Workload','Phase'], columns='Policy', values='Final_Value')
        if not {REACT,GATE,NORM} <= set(p.columns): continue
        rr=(p[REACT]/p[NORM]).mean(); gg=(p[GATE]/p[NORM]).mean()
        print(f"       {s}: reactive {rr:.3f}  gate {gg:.3f}  (gate-adv {(rr-gg)/rr*100:+.2f}%)")
PY

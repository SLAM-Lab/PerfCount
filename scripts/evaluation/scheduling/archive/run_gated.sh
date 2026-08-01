#!/usr/bin/env bash
# P2: does margin-gating rescue the hybrid crossover correction?
#
# Run C showed the raw hybrid is a null under real transition costs (EDP +0.55% with a
# LOSING 44/52 record, ED2P -0.12%). The diagnosis was that its extra E-core residency buys
# migrations whose warmup ED2P charges quadratically. The gate discards the low-claim
# redirects, which should cut that bill.
#
# Both metrics accept the same 6.433% of samples. A fixed margin is NOT comparable across
# metrics -- at margin 0.05 it accepted 55% of flips under EDP but 82% under ED2P, so the
# ED2P arm was barely gated at all and a negative result there would have been
# uninterpretable. Each metric is read from its own run.
#
# Prediction on record before running: EDP roughly holds near +0.55%, ED2P stays negative.
# Ceiling for ANY per-sample gate, measured with perfect hindsight: +1.41% EDP / +1.25% ED2P.
set -e
cd "$(dirname "$0")"

PY=../../../.venv/bin/python3
RES=../../../results/scheduling
PMU=../../../processed_data_10M/x86_desktop_heterogeneous
HP=$RES/Hetero_precompute
DP=$RES/DVFS_precompute
TRACES=$HP/speedup_full_v2_repaired/granular_phase_traces
VITERBI=$HP/viterbi_cache_hetero
export SIM_WORKERS=${SIM_WORKERS:-30}

SUITES="spec_2017 spec_2026 dacapo_c1"
BENCHES=$( { ls $TRACES | grep -oE 'spec_[0-9]+\.[a-z0-9]+_r' | sort -u
             find $PMU/dacapo_c1 -name 'aligned_dacapo_*_cpu0_phase*.csv' -printf '%f\n' 2>/dev/null \
               | grep -oE 'dacapo_[a-z0-9]+' | sort -u; } | sort -u | paste -sd' ')
EXCL=$( comm -23 <(ls $TRACES | grep -oE 'spec_[0-9]+\.[a-z0-9]+_r|dacapo_[a-z0-9]+' | sort -u) \
                 <(printf '%s\n' $BENCHES | sort -u) | paste -sd, )
NBENCH=$(echo $BENCHES | wc -w)

need () { n=$(ls "$1/speedups_from_P_1.0GHz/" 2>/dev/null | grep -oE 'spec_[0-9]+\.[a-z0-9]+_r|dacapo_[a-z0-9]+' | sort -u | wc -l)
          [ "$n" -ge "$NBENCH" ] || { echo "INCOMPLETE: $1 has $n/$NBENCH" >&2; exit 1; } }

TR_EDP=$HP/cross_proc_translate_gated_edp_10M
FC_EDP=$HP/cross_proc_forecast_gated_edp_10M
TR_ED2P=$HP/cross_proc_translate_gatedm_ed2p_10M
FC_ED2P=$HP/cross_proc_forecast_gatedm_ed2p_10M
for d in $TR_EDP $FC_EDP $TR_ED2P $FC_ED2P; do need $d; done
echo "=== all four gated dirs complete at $NBENCH workloads, SIM_WORKERS=$SIM_WORKERS ==="

sim () {
  FAST_HETERO=1 $PY src/main.py --input_dir $TRACES --output_dir $RES/hetero/$1 \
    --power_mode per_sample --decision_power_mode static --warmup_in_decision --apply_warmup \
    --strict_predictions --viterbi_cache_dir $VITERBI ${EXCL:+--exclude_workloads $EXCL} \
    --cross_freq_p_pred_dir $DP/cross_freq_translate_10M --cross_freq_e_pred_dir $DP/cross_freq_translate_10M \
    --cross_freq_p_forecast_dir $DP/forecast_predictions_10M --cross_freq_e_forecast_dir $DP/forecast_predictions_10M \
    --cross_proc_pred_dir $2 --cross_proc_forecast_dir $3
}

echo "=== gated EDP arm (read EDP rows from this run only) ==="
sim gated_edp  $TR_EDP  $FC_EDP
echo "=== gated ED2P arm (read ED2P rows from this run only) ==="
sim gated_ed2p $TR_ED2P $FC_ED2P
echo "=== P2 COMPLETE ==="

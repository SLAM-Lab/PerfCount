#!/usr/bin/env bash
# Wait for the bothcore6 SPEC2017 models, then precompute and score them in the
# decision-relevant (ratio) domain against the top4 and general-temporal baselines.
#
# SPEC2017 first because it is the smaller suite and gives the early read on whether the
# both-core counter set (AUC 0.734 on the crossover) actually lifts ratio correlation above
# top4's 0.195. If it does not, there is no point waiting for the other five combinations.
set -e
cd "$(dirname "$0")"

PY=../../../.venv/bin/python3
O=../../../results/cross_platform/cross_proc/x86_10M
HP=../../../results/scheduling/Hetero_precompute
PMU=../../../processed_data_10M/x86_desktop_heterogeneous
TRACES=$HP/speedup_full_v2_repaired/granular_phase_traces
N_EXPECT=${N_EXPECT:-352}

count () { find "$1" -name '*.cbm' 2>/dev/null | wc -l; }
while :; do
  a=$(count $O/cpu0_to_cpu16/spec_2017/bothcore6)
  b=$(count $O/cpu16_to_cpu0/spec_2017/bothcore6)
  [ "$a" -ge "$N_EXPECT" ] && [ "$b" -ge "$N_EXPECT" ] && break
  pgrep -f cross_proc_x86.py >/dev/null || { echo "TRAINING STOPPED (p2e=$a e2p=$b of $N_EXPECT)"; exit 1; }
  sleep 60
done
echo "=== bothcore6 SPEC2017 models ready (p2e=$a e2p=$b) ==="

$PY cross_proc_precompute.py --model_dir $O --pmu_dir $PMU --oracle_dir $TRACES \
    --out_dir $HP/cross_proc_translate_bothcore6_10M --feature_set bothcore6 --suites spec_2017

echo "=== ratio-domain comparison (SPEC2017) ==="
$PY utils/ratio_accuracy.py \
    $HP/cross_proc_translate_10M \
    $HP/cross_proc_translate_gentemporal_10M \
    $HP/cross_proc_translate_bothcore6_10M \
    --limit 45 --per_config

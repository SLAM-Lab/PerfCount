#!/usr/bin/env bash
# NEED-6, properly this time: does the crossover classifier correction add anything?
#
# The previous attempt was void. gen_hybrid_predictions.py resolved its trace and counter
# paths relative to the CWD, so from this directory every lookup missed, every file was
# written through uncorrected, and the run exited 0 having produced an exact copy of its
# input. The "hybrid" directory was byte-identical to the general-temporal predictions, so
# run C measured a training-mode difference and nothing else. The generator now anchors its
# paths to the repo root and refuses to emit an uncorrected set.
#
# BASE MODEL: general-temporal, not top4-LOOCV. In the decision-relevant (ratio) domain
# gentemporal carries corr 0.604 against top4's 0.195, so correcting top4 would mostly
# measure how much of that gap a classifier can paper over. The question worth answering is
# whether the correction adds anything ON TOP OF the best model available.
#
# Both the translate and forecast directories are corrected. Correcting only one would mix a
# corrected reactive path with an uncorrected forecast path and make any result
# unattributable.
set -e
cd "$(dirname "$0")"

PY=../../../.venv/bin/python3
RES=../../../results/scheduling
PMU=../../../processed_data_10M/x86_desktop_heterogeneous
HP=$RES/Hetero_precompute
DP=$RES/DVFS_precompute
TRACES=$HP/speedup_full_v2_repaired/granular_phase_traces
VITERBI=$HP/viterbi_cache_hetero
export SIM_WORKERS=${SIM_WORKERS:-40}

BASE_TR=$HP/cross_proc_translate_gentemporal_10M
BASE_FC=$HP/cross_proc_forecast_gentemporal_10M
HYB_TR=$HP/cross_proc_translate_realhybrid_10M
HYB_FC=$HP/cross_proc_forecast_realhybrid_10M

# --- 1. generate, refusing to proceed on an uncorrected set --------------------------
# Count files rather than test for the directory: an aborted generation leaves the output
# directory present but incomplete, and a directory test would then skip regeneration and
# simulate a partial or empty prediction set.
N_IN=$(find $BASE_TR -name '*.csv' | wc -l)
gen () {  # $1=in $2=out
  n=$(find "$2" -name '*.csv' 2>/dev/null | wc -l)
  if [ "$n" -lt "$N_IN" ]; then
    echo "generating $2 ($n/$N_IN present)"
    rm -rf "$2"
    $PY gen_hybrid_predictions.py --in_dir "$1" --out_dir "$2" --folds 4
  fi
}
gen $BASE_TR $HYB_TR
gen $BASE_FC $HYB_FC

# --- 2. prove the output is NOT a copy of its input ----------------------------------
$PY - "$BASE_TR" "$HYB_TR" <<'EOF'
import sys, glob, os, numpy as np, pandas as pd
base, hyb = sys.argv[1], sys.argv[2]
fs = sorted(glob.glob(f'{base}/speedups_from_P_3.0GHz/spec_*.csv'))[:25]
diff = tot = 0
for f in fs:
    h = f'{hyb}/speedups_from_P_3.0GHz/{os.path.basename(f)}'
    if not os.path.exists(h):
        sys.exit(f'missing corrected file: {h}')
    b, c = pd.read_csv(f), pd.read_csv(h)
    n = min(len(b), len(c))
    cols = [x for x in b.columns if x.startswith('Speedup_')]
    d = (b[cols].values[:n] != c[cols].values[:n]).any(axis=1)
    diff += int(d.sum()); tot += n
pct = diff / max(tot, 1) * 100
print(f'corrected rows: {diff}/{tot} ({pct:.3f}%) over {len(fs)} files')
if diff == 0:
    sys.exit('ABORT: corrected set is identical to its input. This is the previous defect.')
if pct > 60:
    sys.exit(f'ABORT: {pct:.1f}% of rows changed. A crossover correction should fire on a '
             f'minority of samples; this looks like a different model, not a correction.')
EOF

# --- 3. simulate ---------------------------------------------------------------------
SUITES="spec_2017 spec_2026 dacapo_c1"
BENCHES=$( { ls $TRACES | grep -oE 'spec_[0-9]+\.[a-z0-9]+_r' | sort -u
             find $PMU/dacapo_c1 -name 'aligned_dacapo_*_cpu0_phase*.csv' -printf '%f\n' 2>/dev/null \
               | grep -oE 'dacapo_[a-z0-9]+' | sort -u; } | sort -u | paste -sd' ')
EXCL=$( comm -23 <(ls $TRACES | grep -oE 'spec_[0-9]+\.[a-z0-9]+_r|dacapo_[a-z0-9]+' | sort -u) \
                 <(printf '%s\n' $BENCHES | sort -u) | paste -sd, )

sim () {
  FAST_HETERO=1 $PY src/main.py --input_dir $TRACES --output_dir $RES/hetero/$1 \
    --power_mode per_sample --decision_power_mode static --warmup_in_decision --apply_warmup \
    --strict_predictions --viterbi_cache_dir $VITERBI ${EXCL:+--exclude_workloads $EXCL} \
    --cross_freq_p_pred_dir $DP/cross_freq_translate_10M --cross_freq_e_pred_dir $DP/cross_freq_translate_10M \
    --cross_freq_p_forecast_dir $DP/forecast_predictions_10M --cross_freq_e_forecast_dir $DP/forecast_predictions_10M \
    --cross_proc_pred_dir $2 --cross_proc_forecast_dir $3
}

echo "=== NEED-6: corrected vs gentemporal baseline ==="
sim realhybrid $HYB_TR $HYB_FC
echo "=== compare (baseline arm 'gentemporal' already exists) ==="
RES=$RES $PY compare_runs.py hetero gentemporal realhybrid

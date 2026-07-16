#!/bin/bash
# run_forecasting_under_heterogeneity.sh
# =====================================
# Regenerates every result behind Chapter 5's "Forecasting Under Heterogeneity"
# section, and the figures that report them.
#
# Two independent studies, in the order the chapter presents them:
#
#   TRAIN  (Sec. Training-Time Heterogeneity)
#          Perturbation sweep: replace a fraction p of each training trace's rows
#          with donor rows from another operating point, naive vs translated,
#          for both cores and both donor pools.
#          -> results/forecasting/logs_10M/..._het_{cross_freq,cross_proc}_{naive,translated}
#          -> figures results_cross_{frequency,processor}[_ecore].pdf
#
#   INFER  (Sec. Inference-Time Heterogeneity)
#          Cross-config sweep: translate-then-forecast over all 56 ordered
#          source->target config pairs, with ref_cycles only and with cpu_cycles
#          additionally translated across cores.
#          -> results/forecasting/cross_config/cross_config_10M_modelcmp{,_refcpu}.csv
#          -> figure results_inference_summary.pdf
#
# This script does NOT touch the scheduling simulator. The sim consumes its own
# prediction dumps (run_dump_dvfs.sh -> cap_predictions.py -> run_x86_sweep.sh);
# nothing here feeds it.
#
# PREREQUISITE -- read this before running:
#   Both studies read the CatBoost translators under results/cross_platform/.
#   They must be NEWER than any result you keep, or you regenerate stale numbers.
#   --check verifies this and exits; it is also run automatically before any stage.
#
# Usage:
#   ./run_forecasting_under_heterogeneity.sh --check          # prereqs only, run nothing
#   ./run_forecasting_under_heterogeneity.sh                  # everything
#   ./run_forecasting_under_heterogeneity.sh --only infer     # one stage
#   ./run_forecasting_under_heterogeneity.sh --only figs      # replot from existing CSVs/logs
#   MODELS="dt" ./run_forecasting_under_heterogeneity.sh --only infer   # quick DT-only check
#   DRY=1 ./run_forecasting_under_heterogeneity.sh            # print commands, run nothing
#
# Env:
#   MODELS      models for the inference sweep (default: dt mlp lstm transformer)
#   HORIZONS    forecast horizons for the inference sweep (default: 1)
#   TIMESTEPS   history window for the inference sweep (default: 5)
#   MAX_WORKERS forwarded to the training-sweep launchers
#   DRY=1       print, do not execute

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"      # .../heterogeneous_forecasting/
WF_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"                          # workload_forecasting/
REPO_ROOT="$(cd "$WF_DIR/../../.." && pwd)"
PY="$REPO_ROOT/.venv/bin/python3"

# Both studies now live beside this driver.
TRAIN_DIR="$SCRIPT_DIR/training"
INFER_DIR="$SCRIPT_DIR/inference"
PLOT_TRAIN_DIR="$REPO_ROOT/plotting_scripts/forecasting/heterogeneous_training"
PLOT_INFER_DIR="$REPO_ROOT/plotting_scripts/forecasting/heterogeneous_inference"
FIG_DIR="$REPO_ROOT/figures"

MODELS="${MODELS:-dt mlp lstm transformer}"
HORIZONS="${HORIZONS:-1}"
TIMESTEPS="${TIMESTEPS:-5}"
DRY="${DRY:-0}"
ONLY="all"

while [ $# -gt 0 ]; do
  case "$1" in
    --only) ONLY="$2"; shift 2 ;;
    --check) ONLY="check"; shift ;;
    -h|--help) sed -n '2,50p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1"; exit 2 ;;
  esac
done

run() {  # echo, then execute (or not, under DRY)
  echo "+ $*"
  [ "$DRY" = 1 ] && return 0
  "$@"
}

hr() { printf '=%.0s' {1..72}; echo; }

# ---------------------------------------------------------------- prereqs
CF_TREE="$REPO_ROOT/results/cross_platform/cross_freq/x86_10M"
CP_TREE="$REPO_ROOT/results/cross_platform/cross_proc/x86_10M"
CT_TREE="$CP_TREE/counter_translation"

newest() { find "$1" -name '*.cbm' -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1; }

check_prereqs() {
  hr; echo "PREREQUISITES"; hr
  local fail=0

  for d in "$CF_TREE" "$CP_TREE" "$CT_TREE"; do
    if [ ! -d "$d" ]; then echo "  MISSING  $d"; fail=1; else
      local n; n=$(find "$d" -name '*.cbm' 2>/dev/null | wc -l)
      echo "  ok       $(basename "$(dirname "$d")")/$(basename "$d")  ($n models)"
    fi
  done

  # Coverage: a missing translator is SILENT -- the harness skips the pair and the
  # suite vanishes from the average rather than erroring. Check it explicitly.
  echo
  echo "  cross-frequency translator coverage (cpu0/top4, benches per freq pair):"
  for s in spec_2017 spec_2026 dacapo_c1; do
    local c; c=$(ls "$CF_TREE/cpu0/$s/top4/1.0GHz_to_2.0GHz/"*.cbm 2>/dev/null | wc -l)
    printf "    %-12s %3d\n" "$s" "$c"
    [ "$c" -eq 0 ] && { echo "      ^ no models: this suite will be ABSENT from the cross-frequency average"; fail=1; }
  done

  echo
  echo "  cpu_cycles counter translators (needed for the +cpu_cycles arm):"
  for dir in cpu0_to_cpu16 cpu16_to_cpu0; do
    local c; c=$(find "$CT_TREE/$dir" -name '*.cbm' 2>/dev/null | wc -l)
    printf "    %-16s %4d\n" "$dir" "$c"
    [ "$c" -eq 0 ] && { echo "      ^ absent: that direction silently falls back to copying cpu_cycles"; fail=1; }
  done

  # Staleness: results older than the models they were built from are wrong.
  echo
  echo "  staleness (results older than the newest translator are stale):"
  local m; m=$(newest "$REPO_ROOT/results/cross_platform" | cut -d' ' -f1)
  if [ -n "$m" ]; then
    echo "    newest translator : $(date -d "@${m%.*}" '+%Y-%m-%d %H:%M')"
    for f in "$REPO_ROOT/results/forecasting/cross_config/cross_config_10M_modelcmp_refcpu.csv"; do
      [ -f "$f" ] || continue
      local t; t=$(stat -c %Y "$f")
      if [ "${m%.*}" -gt "$t" ]; then
        echo "    STALE             : $(basename "$f")  ($(date -d "@$t" '+%Y-%m-%d %H:%M'))"
      else
        echo "    current           : $(basename "$f")"
      fi
    done
  fi

  echo
  [ "$fail" = 0 ] && echo "  -> prerequisites OK" || echo "  -> PROBLEMS ABOVE. Results generated now will inherit them."
  return 0
}

# ------------------------------------------------------- TRAIN-time study
stage_train() {
  hr; echo "TRAIN-TIME HETEROGENEITY  (perturbation sweep, p = 0.2 .. 1.0)"; hr
  echo "Each driver runs naive+translated for both cores, sequentially."
  run bash "$TRAIN_DIR/run_het_cross_freq_sweeps.sh" ${MAX_WORKERS:+--max_workers "$MAX_WORKERS"}
  run bash "$TRAIN_DIR/run_het_cross_proc_sweeps.sh" ${MAX_WORKERS:+--max_workers "$MAX_WORKERS"}

  # The sweeps only write per-job logs. Condensing them into the CSVs the figures
  # read is a separate pass, and skipping it leaves the plots on stale data.
  echo
  echo "--- condensing sweep logs into the per-problem CSVs"
  run bash "$TRAIN_DIR/run_het_cross_freq_sweeps.sh" --condense
  run bash "$TRAIN_DIR/run_het_cross_proc_sweeps.sh" --condense

  # The p=0 anchor of every curve is regenerated from the same logs. It is not a
  # separate baseline sweep, so it must be refreshed whenever the sweeps are.
  echo
  echo "--- regenerating the p=0 baseline anchor"
  run "$PY" "$PLOT_TRAIN_DIR/gen_baseline_p0.py"
}

# ------------------------------------------------------- INFER-time study
stage_infer() {
  hr; echo "INFERENCE-TIME HETEROGENEITY  (cross-config, 56 ordered pairs)"; hr
  echo "Two arms: ref_cycles alone, then cpu_cycles additionally across cores."
  echo "The second arm writes the _refcpu CSV; both are needed for the figure."

  # One arm per translated counter set. TRANSLATE picks the output name:
  # a set containing cpu_cycles gets the _refcpu suffix.
  local arm
  for arm in "ref_cycles" "ref_cycles cpu_cycles"; do
    echo
    echo "--- arm: TRANSLATE=\"$arm\"  ->  cross_config_10M_modelcmp$(
        echo "$arm" | grep -q cpu_cycles && echo _refcpu).csv"
    MODELS="$MODELS" HORIZONS="$HORIZONS" TIMESTEPS="$TIMESTEPS" \
      VARIANT=modelcmp TRANSLATE="$arm" \
      run bash "$INFER_DIR/run_cross_config.sh"
  done
}

# ------------------------------------------------------------------ figures
stage_figs() {
  hr; echo "FIGURES"; hr
  run "$PY" "$PLOT_TRAIN_DIR/plot_translation_results.py"
  run "$PY" "$PLOT_INFER_DIR/plot_inference_summary.py"

  # The chapter's \includegraphics points at figures/; the plotters write beside
  # themselves. Copy only what Chapter 5 actually references.
  mkdir -p "$FIG_DIR"
  for spec in "$PLOT_TRAIN_DIR:results_cross_frequency.pdf" \
              "$PLOT_TRAIN_DIR:results_cross_processor.pdf" \
              "$PLOT_INFER_DIR:results_inference_summary.pdf"; do
    d="${spec%%:*}"; f="${spec##*:}"
    if [ -f "$d/$f" ]; then
      run cp "$d/$f" "$FIG_DIR/$f"
    else
      echo "  WARN: $f not produced"
    fi
  done
}

# ---------------------------------------------------------------------- main
echo "Forecasting under heterogeneity — regeneration"
echo "  repo   : $REPO_ROOT"
echo "  stage  : $ONLY"
echo "  models : $MODELS   H: $HORIZONS   T: $TIMESTEPS"
[ "$DRY" = 1 ] && echo "  DRY RUN — nothing will execute"
echo

check_prereqs
[ "$ONLY" = check ] && exit 0
echo

case "$ONLY" in
  all)    stage_train; stage_infer; stage_figs ;;
  train)  stage_train ;;
  infer)  stage_infer ;;
  figs)   stage_figs ;;
  *) echo "unknown stage: $ONLY (want: train|infer|figs|all|check)"; exit 2 ;;
esac

hr
echo "Done. Verify before trusting the numbers:"
echo "  $PY $INFER_DIR/analyze_cross_config.py"
echo "  and re-check that every suite is present in the cross-frequency average."

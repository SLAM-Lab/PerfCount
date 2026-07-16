#!/bin/bash
# setup_simulator.sh -- build every prediction set the scheduling simulator consumes.
#
# The simulator itself reads only finished CSVs. Everything it needs is produced by this
# script, in dependency order, from the oracle traces plus the trained cross-platform
# models. Run this to completion and run_dvfs_study.sh / run_x86_sweep.sh will launch.
#
# STAGES (each builds one group of prediction dirs under results/scheduling/):
#
#   translate  cross_freq_translate_10M      cross_freq_precompute.py, per core
#              cross_proc_translate_10M      cross_proc_precompute.py        [XPROC=1]
#                 Same-chunk model translation. Row i predicts chunk i from chunk i's own
#                 counters. Feeds the REACTIVE model policies.
#
#   forecast   forecast_predictions_10M      run_dump_dvfs.sh, method=per_phase
#              forecast_unaware_10M          run_dump_dvfs.sh, method=global
#              cross_proc_forecast_10M       run_dump_dvfs.sh, XPROC=1        [XPROC=1]
#              cross_proc_forecast_unaware_10M                                [XPROC=1]
#                 Walk-forward causal forecast. Row i is a forecast OF chunk i built from
#                 history through i-1. Feeds the FORECASTING model policies.
#
#   oracle     forecast_oracle_10M           run_dump_forecast_oracle.sh, per_phase
#              forecast_oracle_unaware_10M   run_dump_forecast_oracle.sh, global
#                 Forecast x Oracle bound: each config forecasts its OWN true series, so
#                 translation is the identity. Isolates forecast error from translation
#                 error. A bound, not a deployable policy.
#
#   gated      forecast_gated_10M            run_dump_dvfs.sh, gate=persist
#              forecast_oracle_gated_10M     run_dump_forecast_oracle.sh, gate=persist
#                 Forecast with a rolling persistence gate: fall back to the lagged
#                 persistence forecast when the model has been losing to it.
#
#   cap        capped/cf_{tr,fc}_cap{5,10,20}   cap_predictions.py
#              capped/cp_{tr,fc}_cap{5,10,20}                                 [XPROC=1]
#                 Clamps predicted speedups to within +/-N% of truth. Feeds the
#                 model-accuracy sensitivity study. Depends on translate + forecast.
#
#   verify     Coverage check only. Never builds.
#
# IDEMPOTENT. Every stage verifies its outputs against the trace set first and skips a dir
# that is already complete. To rebuild, either delete the dir or name its stage in FORCE.
# This matters: the recurring defect in this pipeline is a plausible default silently
# standing in for missing data (a stale suite dir, a 10 W constant, an oracle fallback), so
# a partial dump that looks finished is the failure mode to design against. Coverage is
# checked by stem against the traces, not by counting files.
#
#   ./setup_simulator.sh                        # bring everything up to date
#   DRY=1 ./setup_simulator.sh                  # print the plan, build nothing
#   STAGES=verify ./setup_simulator.sh          # audit coverage only
#   FORCE=forecast STAGES="forecast cap" ./setup_simulator.sh
#   XPROC=1 ./setup_simulator.sh                # include cross-processor (needs the retrain)
#   BENCHES="dacapo_avrora spec_521.wrf_r" STAGES=forecast ./setup_simulator.sh   # smoke
#
# Env: STAGES FORCE DRY PAR XPROC CAPS BENCHES TRACES MODEL SUITES
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCHED_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$SCHED_DIR/../../.." && pwd)"
PY="$REPO_ROOT/.venv/bin/python3"
RES="$REPO_ROOT/results/scheduling"
PF="$REPO_ROOT/scripts/evaluation/workload_forecasting/phase_forecasting"

TRACES="${TRACES:-$RES/speedup_full_v2/granular_phase_traces}"
PMU="${PMU:-$REPO_ROOT/processed_data_10M/x86_desktop_heterogeneous}"
CF_MODELS="${CF_MODELS:-$REPO_ROOT/results/cross_platform/cross_freq/x86_10M}"
CP_MODELS="${CP_MODELS:-$REPO_ROOT/results/cross_platform/cross_proc/x86_10M}"
# cross_freq_precompute resolves <model_base_dir>/<cpu>/<suite>/full. It WARNS and continues
# when a suite dir is absent, which is how a suite goes silently missing. Named explicitly
# here, and the verify stage is the backstop that turns such a gap into a hard failure.
SUITES="${SUITES:-spec_2017 spec_2026 dacapo_c2}"

STAGES="${STAGES:-translate forecast oracle gated cap}"
FORCE="${FORCE:-}"
DRY="${DRY:-0}"
XPROC="${XPROC:-0}"
PAR="${PAR:-24}"
CAPS="${CAPS:-5 10 20}"
MODEL="${MODEL:-dt}"
CORES="0 16"; FREQS="1.0 2.0 3.0 4.0"

forced(){ [ "$FORCE" = all ] && return 0; for f in $FORCE; do [ "$f" = "$1" ] && return 0; done; return 1; }
staged(){ for s in $STAGES; do [ "$s" = "$1" ] && return 0; done; return 1; }

# ---------------------------------------------------------------- preflight
fail=0
[ -x "$PY" ]        || { echo "FATAL: no python at $PY"; fail=1; }
[ -d "$TRACES" ]    || { echo "FATAL: traces not found: $TRACES"; fail=1; }
staged translate && { [ -d "$PMU" ] || { echo "FATAL: pmu dir not found: $PMU"; fail=1; }
                      [ -d "$CF_MODELS" ] || { echo "FATAL: cross-freq models not found: $CF_MODELS"; fail=1; }; }
[ "$fail" = 0 ] || { echo; echo "Preflight failed. Nothing run."; exit 1; }

TMPD="$(mktemp -d)"; trap 'rm -rf "$TMPD"' EXIT
# Build logs outlive the run. A multi-hour dump that fails at hour three is unbudgetable to
# repeat blind, so the log has to survive the trap above.
LOGDIR="${LOGDIR:-$RES/setup_logs}"; mkdir -p "$LOGDIR"

# The bench-phase set the SIMULATOR reads, taken from the traces it is actually given.
# Every prediction dir is checked against exactly this list. Deriving it from the trace dir
# rather than hardcoding it is what makes a missing suite or a dropped bench visible.
ls "$TRACES" | sed -nE 's/^speedups_P_4\.0GHz_(.+)\.csv$/\1/p' | sort -u > "$TMPD/stems"
# An explicit BENCHES is a smoke test. Narrow the expected set to it, otherwise every dir
# reports MISS against the full 147 and the check is useless exactly when it is cheapest.
if [ -n "${BENCHES:-}" ]; then
  : > "$TMPD/sel"; for b in $BENCHES; do grep -xE "$(printf '%s' "$b" | sed 's/[.[\*^$]/\\&/g')_phase[0-9]+" "$TMPD/stems" >> "$TMPD/sel"; done
  sort -u "$TMPD/sel" -o "$TMPD/stems"
  [ -s "$TMPD/stems" ] || { echo "FATAL: BENCHES matched no trace bench-phases: $BENCHES"; exit 1; }
fi
NPH=$(wc -l < "$TMPD/stems")
NB=$(sed -E 's/_phase[0-9]+$//' "$TMPD/stems" | sort -u | wc -l)
[ "$NPH" -gt 0 ] || { echo "FATAL: no bench-phases found in $TRACES"; exit 1; }
BENCHES="${BENCHES:-$(sed -E 's/_phase[0-9]+$//' "$TMPD/stems" | sort -u)}"
NEXP=$(( 8 * NPH ))   # 8 source configs (2 cores x 4 freqs) x bench-phases

echo "=== setup_simulator"
echo "    traces : $TRACES"
echo "    scope  : $NB benches / $NPH bench-phases / 8 source configs -> $NEXP files per dir"
echo "    stages : $STAGES${FORCE:+   (force: $FORCE)}"
echo "    xproc  : $([ "$XPROC" = 1 ] && echo enabled || echo 'skipped (set XPROC=1)')"
[ "$DRY" = 1 ] && echo "    DRY RUN -- nothing will be built"
echo

# ---------------------------------------------------------------- coverage check
# $1=dir  $2=layout: cf (speedups_from_C_FGHz/speedups_C_FGHz_<stem>.csv)
#                    cp (speedups_from_C_FGHz/<stem>.csv)
#                    fo (C_FGHz/<stem>.csv)
# Sets MISSING to the shortfall. Returns 0 only when every config holds every stem.
check_dir(){ local dir=$1 lay=$2 c f sub want got
  MISSING=0; MISSDESC=""
  [ -d "$dir" ] || { MISSING=$NEXP; MISSDESC="dir absent"; return 1; }
  for c in P E; do for f in $FREQS; do
    case $lay in
      cf|cp) sub="$dir/speedups_from_${c}_${f}GHz" ;;
      fo)    sub="$dir/${c}_${f}GHz" ;;
    esac
    if [ ! -d "$sub" ]; then MISSING=$((MISSING+NPH)); MISSDESC="${MISSDESC}${MISSDESC:+, }no ${c}_${f}GHz"; continue; fi
    case $lay in
      cf) ls "$sub" | sed -nE "s/^speedups_${c}_${f}GHz_(.+)\.csv$/\1/p" ;;
      *)  ls "$sub" | sed -nE 's/^(.+)\.csv$/\1/p' ;;
    esac | sort -u > "$TMPD/got"
    got=$(comm -23 "$TMPD/stems" "$TMPD/got" | wc -l)
    if [ "$got" -gt 0 ]; then
      MISSING=$((MISSING+got))
      MISSDESC="${MISSDESC}${MISSDESC:+, }${c}_${f}GHz -$got"
    fi
  done; done
  [ "$MISSING" = 0 ]
}

report(){ local dir=$1 lay=$2
  if check_dir "$dir" "$lay"; then
    printf '  %-34s OK    %d/%d\n' "$(basename "$dir")" "$NEXP" "$NEXP"
  else
    printf '  %-34s MISS  %d/%d   (%s)\n' "$(basename "$dir")" "$((NEXP-MISSING))" "$NEXP" "$MISSDESC"
    return 1
  fi
}

# $1=dir $2=layout $3=stage $4..=build command. Skips a complete dir unless its stage is forced.
build(){ local dir=$1 lay=$2 stage=$3; shift 3
  if check_dir "$dir" "$lay" && ! forced "$stage"; then
    printf '  %-34s complete, skipping\n' "$(basename "$dir")"; return 0
  fi
  if forced "$stage" && [ -d "$dir" ]; then
    echo "  $(basename "$dir"): FORCE -> removing and rebuilding"
    [ "$DRY" = 1 ] || rm -rf "$dir"
  elif [ -d "$dir" ]; then
    echo "  $(basename "$dir"): incomplete ($MISSING missing) -> building"
  else
    echo "  $(basename "$dir"): building"
  fi
  if [ "$DRY" = 1 ]; then echo "      $*" | tr -s ' '; return 0; fi
  local t0=$SECONDS lg="$LOGDIR/$(basename "$dir").log"
  if "$@" > "$lg" 2>&1; then
    if check_dir "$dir" "$lay"; then
      echo "      done in $(( (SECONDS-t0)/60 ))m   ($lg)"
    else
      echo "      BUILT BUT INCOMPLETE: $MISSING/$NEXP missing ($MISSDESC)"; tail -15 "$lg"; return 1
    fi
  else
    echo "      FAILED -- see $lg"; tail -20 "$lg"; return 1
  fi
}

dump_cf(){ # $1=out $2=method $3=gate $4=xproc
  OUT_DIR="$1" METHOD="$2" GATE="$3" XPROC="$4" MODEL="$MODEL" PAR="$PAR" \
    BENCHES="$BENCHES" CORES="$CORES" FREQS="$FREQS" bash "$PF/run_dump_dvfs.sh"; }
dump_fo(){ # $1=out $2=method $3=gate
  OUT_DIR="$1" METHOD="$2" GATE="$3" MODEL="$MODEL" PAR="$PAR" \
    BENCHES="$BENCHES" CORES="$CORES" FREQS="$FREQS" bash "$PF/run_dump_forecast_oracle.sh"; }
cf_pre(){ # $1=core
  "$PY" "$SCHED_DIR/cross_freq_precompute.py" --model_base_dir "$CF_MODELS" --pmu_dir "$PMU" \
    --oracle_dir "$TRACES" --out_dir "$RES/cross_freq_translate_10M" --core_type "$1" --suites $SUITES; }
cp_pre(){
  "$PY" "$SCHED_DIR/cross_proc_precompute.py" --model_dir "$CP_MODELS" --pmu_dir "$PMU" \
    --oracle_dir "$TRACES" --out_dir "$RES/cross_proc_translate_10M" --suites $SUITES; }
cap(){ # $1=pred_dir $2=out_base
  "$PY" "$SCRIPT_DIR/cap_predictions.py" --pred_dir "$1" --granular "$TRACES" \
    --out_base "$2" --caps $CAPS --workers "$PAR"; }

rc=0

# ---------------------------------------------------------------- stages
if staged translate; then
  echo "--- translate (same-chunk model translation -> reactive policies)"
  # Both cores write into one dir (disjoint speedups_from_{P,E}_* subtrees), so the
  # completeness check only holds after both have run. Force P whenever E is being built.
  if ! check_dir "$RES/cross_freq_translate_10M" cf || forced translate; then
    if [ "$DRY" = 1 ]; then
      echo "  cross_freq_translate_10M: building"
      for c in P E; do echo "      cross_freq_precompute.py --core_type $c --suites $SUITES"; done
    else
      forced translate && rm -rf "$RES/cross_freq_translate_10M"
      for c in P E; do
        echo "  cross_freq_translate_10M: core $c"
        cf_pre "$c" > "$LOGDIR/cross_freq_translate_10M.$c.log" 2>&1 || { echo "      FAILED (core $c)"; tail -20 "$LOGDIR/cross_freq_translate_10M.$c.log"; rc=1; }
      done
      report "$RES/cross_freq_translate_10M" cf || rc=1
    fi
  else
    printf '  %-34s complete, skipping\n' cross_freq_translate_10M
  fi
  [ "$XPROC" = 1 ] && { build "$RES/cross_proc_translate_10M" cp translate cp_pre || rc=1; }
  echo
fi

if staged forecast; then
  echo "--- forecast (walk-forward causal forecast -> forecasting policies)"
  build "$RES/forecast_predictions_10M" cf forecast dump_cf "$RES/forecast_predictions_10M" per_phase none 0 || rc=1
  build "$RES/forecast_unaware_10M"     cf forecast dump_cf "$RES/forecast_unaware_10M"     global   none 0 || rc=1
  if [ "$XPROC" = 1 ]; then
    build "$RES/cross_proc_forecast_10M"         cp forecast dump_cf "$RES/cross_proc_forecast_10M"         per_phase none 1 || rc=1
    build "$RES/cross_proc_forecast_unaware_10M" cp forecast dump_cf "$RES/cross_proc_forecast_unaware_10M" global    none 1 || rc=1
  fi
  echo
fi

if staged oracle; then
  echo "--- oracle (Forecast x Oracle bound: self-forecast, identity translation)"
  build "$RES/forecast_oracle_10M"         fo oracle dump_fo "$RES/forecast_oracle_10M"         per_phase none || rc=1
  build "$RES/forecast_oracle_unaware_10M" fo oracle dump_fo "$RES/forecast_oracle_unaware_10M" global    none || rc=1
  echo
fi

if staged gated; then
  echo "--- gated (forecast with a rolling persistence gate)"
  build "$RES/forecast_gated_10M"        cf gated dump_cf "$RES/forecast_gated_10M" per_phase persist 0 || rc=1
  build "$RES/forecast_oracle_gated_10M" fo gated dump_fo "$RES/forecast_oracle_gated_10M" per_phase persist || rc=1
  # Cross-processor gated arm. The sim already registers Model_Forecast_Gated_Hetero against
  # this dir; only the dump was missing. Gated on the CROSS-PROC forecast, which loses to the
  # heuristic on spec26 raw (2x energy), so the persistence gate is exactly the mechanism that
  # should bound it back. Blocked on the cross-proc retrain -- building it against the current
  # model reproduces the 1GHz-at-2GHz defect -- so it is XPROC-gated like the rest of that arm.
  if [ "$XPROC" = 1 ]; then
    build "$RES/cross_proc_forecast_gated_10M" cp gated dump_cf "$RES/cross_proc_forecast_gated_10M" per_phase persist 1 || rc=1
  fi
  echo
fi

if staged cap; then
  echo "--- cap (error-capped predictions -> model-accuracy sensitivity)"
  # cap_predictions writes <out_base>_cap<N> per cap in one pass, so check the last cap as
  # the sentinel: if it is complete the pass finished.
  last_cap=$(echo $CAPS | tr ' ' '\n' | tail -1)
  for pair in "cross_freq_translate_10M:capped/cf_tr:cf" "forecast_predictions_10M:capped/cf_fc:cf"; do
    src="${pair%%:*}"; rest="${pair#*:}"; base="${rest%%:*}"; lay="${rest##*:}"
    if ! check_dir "$RES/$src" "$lay"; then
      echo "  ${base}_cap*: SKIPPED, source $src is incomplete ($MISSING missing)"; rc=1; continue
    fi
    build "$RES/${base}_cap${last_cap}" "$lay" cap cap "$RES/$src" "$RES/$base" || rc=1
  done
  if [ "$XPROC" = 1 ]; then
    for pair in "cross_proc_translate_10M:capped/cp_tr:cp" "cross_proc_forecast_10M:capped/cp_fc:cp"; do
      src="${pair%%:*}"; rest="${pair#*:}"; base="${rest%%:*}"; lay="${rest##*:}"
      if ! check_dir "$RES/$src" "$lay"; then
        echo "  ${base}_cap*: SKIPPED, source $src is incomplete"; rc=1; continue
      fi
      build "$RES/${base}_cap${last_cap}" "$lay" cap cap "$RES/$src" "$RES/$base" || rc=1
    done
  fi
  echo
fi

# ---------------------------------------------------------------- final audit
echo "=== coverage (every dir checked stem-by-stem against the $NPH trace bench-phases)"
ok=0
report "$RES/cross_freq_translate_10M"    cf || ok=1
report "$RES/forecast_predictions_10M"    cf || ok=1
report "$RES/forecast_unaware_10M"        cf || ok=1
report "$RES/forecast_oracle_10M"         fo || ok=1
report "$RES/forecast_oracle_unaware_10M" fo || ok=1
[ -d "$RES/forecast_gated_10M" ]        && { report "$RES/forecast_gated_10M"        cf || ok=1; }
[ -d "$RES/forecast_oracle_gated_10M" ] && { report "$RES/forecast_oracle_gated_10M" fo || ok=1; }
for n in $CAPS; do
  [ -d "$RES/capped/cf_tr_cap$n" ] && { report "$RES/capped/cf_tr_cap$n" cf || ok=1; }
  [ -d "$RES/capped/cf_fc_cap$n" ] && { report "$RES/capped/cf_fc_cap$n" cf || ok=1; }
done
if [ "$XPROC" = 1 ]; then
  report "$RES/cross_proc_translate_10M" cp || ok=1
  report "$RES/cross_proc_forecast_10M"  cp || ok=1
  [ -d "$RES/cross_proc_forecast_gated_10M" ] && { report "$RES/cross_proc_forecast_gated_10M" cp || ok=1; }
fi

echo
if [ "$ok" = 0 ] && [ "$rc" = 0 ]; then
  echo "=== all prediction sets complete. Launch with:"
  echo "  $SCRIPT_DIR/run_dvfs_study.sh                       # DVFS only"
  echo "  ACC=\"raw cap20 cap10 cap5\" $SCRIPT_DIR/run_dvfs_study.sh   # + accuracy sweep"
  [ "$XPROC" = 1 ] && echo "  $SCHED_DIR/run_x86_sweep.sh                          # full x86 (DVFS + hetero)"
else
  echo "=== INCOMPLETE. The simulator substitutes ORACLE times for any missing prediction,"
  echo "    which reads as a near-perfect model result. Do not launch a study against a dir"
  echo "    marked MISS above. Rebuild it, or pass --strict_predictions and let it fail loud."
  exit 1
fi

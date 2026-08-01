#!/bin/bash
# run_spec_dt_mlp.sh
# ==================
# Chapter 4 multi-counter forecasting: how forecastable are the counters that
# cross-platform prediction relies on? Runs the existing run_multi_counter.py
# harness restricted to DT + MLP on the SPEC suites (2017 + 2026) only, both x86
# cores, all four frequencies.
#
# Targets are the harness default: the top cross-frequency / cross-processor
# counters plus the ref_cycles baseline and the branches (top4b) signal.
#
# SPEC-only, both-core, 4-freq workload set (47 benches) is passed explicitly so
# the run is reproducible and does not depend on the harness default (which
# includes DaCapo).
#
# Usage:
#   ./run_spec_dt_mlp.sh            # both cores, PAR=80
#   PAR=40 ./run_spec_dt_mlp.sh
#   DRY=1 ./run_spec_dt_mlp.sh

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
PY="$REPO/.venv/bin/python3"
PAR="${PAR:-80}"

export MODELS="dt mlp"
# 47 SPEC workloads present on BOTH cores at all four frequencies (21 SPEC2017 + 26 SPEC2026).
export BENCHES="spec_500.perlbench_r spec_502.gcc_r spec_503.bwaves_r spec_505.mcf_r \
spec_508.namd_r spec_510.parest_r spec_511.povray_r spec_519.lbm_r spec_520.omnetpp_r \
spec_521.wrf_r spec_523.xalancbmk_r spec_525.x264_r spec_527.cam4_r spec_531.deepsjeng_r \
spec_538.imagick_r spec_541.leela_r spec_544.nab_r spec_548.exchange2_r spec_549.fotonik3d_r \
spec_554.roms_r spec_557.xz_r spec_706.stockfish_r spec_707.ntest_r spec_708.sqlite_r \
spec_709.cactus_r spec_710.omnetpp_r spec_714.cpython_r spec_721.gcc_r spec_722.palm_r \
spec_723.llvm_r spec_727.cppcheck_r spec_729.abc_r spec_731.astcenc_r spec_734.vpr_r \
spec_735.gem5_r spec_736.ocio_r spec_737.gmsh_r spec_748.flightdm_r spec_749.fotonik3d_r \
spec_750.sealcrypto_r spec_753.ns3_r spec_765.roms_r spec_766.femflow_r spec_767.nest_r \
spec_772.marian_r spec_777.zstd_r spec_782.lbm_r"

nb=$(echo $BENCHES | wc -w)
echo "=== multi-counter forecasting: DT+MLP on SPEC ($nb workloads), both cores ==="
for CPU in 0 16; do
  echo
  echo "---- cpu$CPU ($([ $CPU = 0 ] && echo P-core || echo E-core)) ----"
  [ "${DRY:-0}" = 1 ] && { echo "+ MODELS='$MODELS' BENCHES=<$nb> $PY run_multi_counter.py --cpu $CPU --max_workers $PAR"; continue; }
  "$PY" "$SCRIPT_DIR/run_multi_counter.py" --cpu "$CPU" --max_workers "$PAR"
done
echo
echo "Done. Condensed CSVs: $SCRIPT_DIR/condensed_multi_counter_cpu{0,16}.csv"

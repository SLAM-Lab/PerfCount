"""Multi-counter forecasting sweep for the x86 heterogeneous desktop.

Question: cross-platform prediction consumes a handful of counters (Chapter 4's
cross-frequency and cross-processor importance tables). Can those same counters
be *forecast* one window ahead, so the whole pipeline (forecast the counter, then
translate it across platforms) is viable?

This sweep forecasts each of a set of TARGET counters chosen from those importance
tables (including branches, the top4b signal, and branch_misses, a top
cross-processor signal) from the full counter history, at horizon 1, on both cores
of the x86 desktop. It reuses src/forecasting.py, which forecasts
input_counters[0]; we rotate each target to the front.

  logs   : results/forecasting/logs_10M/multi_counter/{MACHINE}_cpu{CPU}/{target}/
             {freq}GHz/horizon_1/timesteps_{T}/{workload}_{model}.log
  output : condensed_multi_counter_cpu{CPU}.csv   (beside this script)
             columns: core,target,model,frequency,workload,mape,...

Usage:
  python run_multi_counter.py --cpu 0
  python run_multi_counter.py --cpu 16 --condense
  MODELS="dt" python run_multi_counter.py --cpu 0            # fast DT-only pass
  MODELS="dt mlp lstm transformer" TARGETS="branches branch_misses" \
      python run_multi_counter.py --cpu 0
"""
import os
import re
import argparse
import subprocess
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

os.environ.update({
    'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
    'BLIS_NUM_THREADS': '1', 'VECLIB_MAXIMUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1',
    'TF_NUM_INTRAOP_THREADS': '1', 'TF_NUM_INTEROP_THREADS': '1', 'TF_CPP_MIN_LOG_LEVEL': '3',
})

MACHINE = "x86_desktop_heterogeneous"
GRAN    = "10M"
EPOCHS  = 50
HORIZON = 1
TIMESTEPS = int(os.environ.get("TIMESTEPS", "10"))

FREQS  = os.environ.get("FREQS", "1.0 2.0 3.0 4.0").split()
MODELS = os.environ.get("MODELS", "dt mlp lstm transformer").split()

# Target counters, drawn from the Chapter 4 x86 cross-frequency and cross-processor
# importance tables, plus the ref_cycles baseline. All are present on BOTH cores.
#   cross-frequency top signals : cache_misses cache_references llc_loads llc_misses dtlb_load_misses
#   cross-processor top signals : branch_misses branch_load_misses l1_icache_load_misses
#   top4b signal                : branches
#   forecasting baseline        : ref_cycles
TARGETS = os.environ.get("TARGETS",
    "ref_cycles branches branch_misses branch_load_misses cache_misses "
    "cache_references llc_loads llc_misses dtlb_load_misses l1_icache_load_misses").split()

# Per-core full counter set (E-core lacks fp_arith_scalar_single, l1_dcache_load_misses).
COUNTERS_BY_CPU = {
    "0": ("cpu_cycles branch_load_misses branch_loads branch_misses branches bus_cycles "
          "cache_misses cache_references dtlb_load_misses dtlb_loads dtlb_store_misses "
          "dtlb_stores fp_arith_scalar_single instructions itlb_load_misses "
          "l1_dcache_load_misses l1_dcache_loads l1_dcache_stores l1_icache_load_misses "
          "llc_loads llc_misses mem_stores ref_cycles").split(),
    "16": ("cpu_cycles branch_load_misses branch_loads branch_misses branches bus_cycles "
           "cache_misses cache_references dtlb_load_misses dtlb_loads dtlb_store_misses "
           "dtlb_stores instructions itlb_load_misses l1_dcache_loads l1_dcache_stores "
           "l1_icache_load_misses llc_loads llc_misses mem_stores ref_cycles").split(),
}

BENCHES = os.environ.get("BENCHES", (
    "dacapo_avrora dacapo_batik dacapo_biojava dacapo_cassandra dacapo_eclipse dacapo_fop "
    "dacapo_graphchi dacapo_h2 dacapo_h2o dacapo_jme dacapo_jython dacapo_kafka dacapo_luindex "
    "dacapo_lusearch dacapo_pmd dacapo_spring dacapo_sunflow dacapo_tomcat dacapo_tradebeans "
    "dacapo_tradesoap dacapo_xalan dacapo_zxing spec_500.perlbench_r spec_502.gcc_r "
    "spec_503.bwaves_r spec_505.mcf_r spec_507.cactuBSSN_r spec_508.namd_r spec_510.parest_r "
    "spec_511.povray_r spec_519.lbm_r spec_520.omnetpp_r spec_521.wrf_r spec_523.xalancbmk_r "
    "spec_525.x264_r spec_527.cam4_r spec_531.deepsjeng_r spec_538.imagick_r spec_541.leela_r "
    "spec_544.nab_r spec_548.exchange2_r spec_549.fotonik3d_r spec_554.roms_r spec_557.xz_r "
    "spec_707.ntest_r spec_708.sqlite_r spec_709.cactus_r spec_710.omnetpp_r spec_714.cpython_r "
    "spec_721.gcc_r spec_722.palm_r spec_723.llvm_r spec_727.cppcheck_r spec_729.abc_r "
    "spec_731.astcenc_r spec_734.vpr_r spec_735.gem5_r spec_736.ocio_r spec_737.gmsh_r "
    "spec_748.flightdm_r spec_749.fotonik3d_r spec_750.sealcrypto_r spec_753.ns3_r "
    "spec_765.roms_r spec_766.femflow_r spec_767.nest_r spec_777.zstd_r spec_782.lbm_r").split())

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WF_DIR     = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
ROOT_DIR   = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../../results/forecasting"))
run_env    = dict(os.environ)
run_env["PYTHONPATH"] = WF_DIR + os.pathsep + run_env.get("PYTHONPATH", "")

_OK = re.compile(r"'mape'")


def log_dir(cpu, target, freq):
    return os.path.join(ROOT_DIR, f"logs_{GRAN}", "multi_counter",
                        f"{MACHINE}_cpu{cpu}", target, f"{freq}GHz",
                        f"horizon_{HORIZON}", f"timesteps_{TIMESTEPS}")


def is_ok(log_file):
    try:
        with open(log_file, errors="ignore") as f:
            return _OK.search(f.read()) is not None
    except OSError:
        return False


def jobs_for(cpu):
    return [(cpu, tgt, freq, m, wl)
            for tgt in TARGETS for freq in FREQS for m in MODELS for wl in BENCHES]


def run_job(job):
    cpu, target, freq, m, bench = job
    counters = COUNTERS_BY_CPU[cpu]
    if target not in counters:
        return
    ordered = [target] + [c for c in counters if c != target]   # target -> index 0
    workload = f"aligned_{bench}_{freq}GHz_cpu{cpu}"
    ldir = log_dir(cpu, target, freq)
    os.makedirs(ldir, exist_ok=True)
    log_file = os.path.join(ldir, f"{workload}_{m}.log")
    if is_ok(log_file):
        return
    cmd = ["python3", "src/forecasting.py", "--benchmark", workload,
           "--dataset", MACHINE, "--input_counters", *ordered,
           "--model", m, "--timesteps", str(TIMESTEPS), "--forecast_horizon", str(HORIZON),
           "--epochs", str(EPOCHS), "--batch_size", "32", "--neurons", "16",
           "--optimizer", "nadam", "--loss_function", "mae", "--name", f"{workload}_{m}"]
    with open(log_file, "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=WF_DIR, env=run_env)


def condense(cpu):
    rows = []
    for cpu_, target, freq, m, bench in jobs_for(cpu):
        if target not in COUNTERS_BY_CPU[cpu_]:
            continue
        workload = f"aligned_{bench}_{freq}GHz_cpu{cpu_}"
        log_file = os.path.join(log_dir(cpu_, target, freq), f"{workload}_{m}.log")
        rec = {"core": f"cpu{cpu_}", "target": target, "model": m,
               "frequency": freq, "workload": workload,
               "status": "Success" if is_ok(log_file) else "Failed/Missing"}
        if rec["status"] == "Success":
            with open(log_file, errors="ignore") as f:
                for k, v in re.findall(
                        r"'([^']+)'\s*:\s*(?:np\.float64\()?([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\)?",
                        f.read()):
                    rec[k] = float(v)
        rows.append(rec)
    out = os.path.join(SCRIPT_DIR, f"condensed_multi_counter_cpu{cpu}.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    ok = sum(r["status"] == "Success" for r in rows)
    print(f"Condensed {ok}/{len(rows)} successful jobs -> {out}")


def main():
    ap = argparse.ArgumentParser(description="Multi-counter forecasting sweep (x86 desktop).")
    ap.add_argument("--cpu", required=True, choices=["0", "16"])
    ap.add_argument("--max_workers", type=int, default=80)
    ap.add_argument("--condense", action="store_true")
    args = ap.parse_args()

    jobs = jobs_for(args.cpu)
    if args.condense:
        condense(args.cpu)
        return
    print(f"--- MULTI-COUNTER SWEEP cpu{args.cpu} : {len(jobs)} jobs "
          f"({len(TARGETS)} targets x {len(FREQS)} freqs x {len(MODELS)} models x "
          f"{len(BENCHES)} benches) ---")
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        list(ex.map(run_job, jobs))
    condense(args.cpu)


if __name__ == "__main__":
    main()

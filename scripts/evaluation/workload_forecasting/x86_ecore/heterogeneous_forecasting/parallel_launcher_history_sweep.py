import os
import glob
import subprocess
import argparse
import re
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

os.environ.update({
    'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
    'BLIS_NUM_THREADS': '1', 'VECLIB_MAXIMUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1',
    'TF_NUM_INTRAOP_THREADS': '1', 'TF_NUM_INTEROP_THREADS': '1'
})

MACHINE   = "x86_desktop_heterogeneous"
CPU       = "16"         # E-Core
MAX_WORKERS = 80
EPOCHS    = 50
HET_SEED  = 42

FREQS                = ["1.0", "2.0", "3.0", "4.0"]
HORIZONS             = [1, 8, 16, 32]
TIMESTEPS            = [10]
MODELS               = ["dt", "mlp", "lstm", "transformer"]
GRANULARITIES        = ["10M"]
HET_PROBS            = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
HETEROGENEOUS_MODES  = ["cross_freq", "cross_proc"]

# x86 E-Core (cpu16) native 21-counter set — also the shared set used for
# cross_proc on both cores, since it's a strict subset of the P-core set
COUNTERS_SHARED = (
    "cpu_cycles branch_load_misses branch_loads branch_misses branches bus_cycles "
    "cache_misses cache_references dtlb_load_misses dtlb_loads dtlb_store_misses "
    "dtlb_stores instructions itlb_load_misses l1_dcache_loads l1_dcache_stores "
    "l1_icache_load_misses llc_loads llc_misses mem_stores ref_cycles"
).split()

COUNTERS = {
    "cross_freq": COUNTERS_SHARED,
    "cross_proc": COUNTERS_SHARED,
}

BENCHMARK_NAME_RE = re.compile(
    r"^aligned_(?P<rest>.+)_(?P<freq>[\d.]+)GHz_cpu(?P<cpu>\d+)_phase(?P<phase>\d+)$"
)
SPEC_BENCH_RE = re.compile(r"^spec_(\d+)\.")

SCRIPT_DIR             = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR               = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../../../results/forecasting/"))
WORKLOAD_FORECASTING_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))

run_env = os.environ.copy()
run_env["PYTHONPATH"] = WORKLOAD_FORECASTING_DIR

def _suite_of(rest):
    if rest.startswith("dacapo_"):
        return "dacapo"
    m = SPEC_BENCH_RE.match(rest)
    if m:
        return "spec2026" if int(m.group(1)) >= 700 else "spec2017"
    return None

def get_all_jobs():
    jobs = []
    for gran in GRANULARITIES:
        data_dir = os.path.abspath(
            os.path.join(SCRIPT_DIR, f"../../../../../processed_data_{gran}/{MACHINE}")
        )
        for freq in FREQS:
            pattern = os.path.join(data_dir, f"aligned_*_{freq}GHz_cpu{CPU}_phase*.csv")
            for fpath in sorted(glob.glob(pattern)):
                name = os.path.basename(fpath).replace('.csv', '')
                bm = BENCHMARK_NAME_RE.match(name)
                if not bm:
                    continue
                suite = _suite_of(bm.group('rest'))
                if suite is None:
                    continue
                for mode in HETEROGENEOUS_MODES:
                    for p in HET_PROBS:
                        for m in MODELS:
                            for h in HORIZONS:
                                for t in TIMESTEPS:
                                    jobs.append((gran, suite, freq, h, t, m, mode, p, name))
    return jobs

def is_job_successful(log_file):
    if not os.path.exists(log_file):
        return False
    with open(log_file, 'r', errors='ignore') as f:
        content = f.read()
        if not content.strip():
            return False
        if "Traceback" in content or "Killed" in content or "Error" in content or "Exception" in content:
            return False
    return True

def _log_path(job_args):
    gran, suite, freq, h, t, m, mode, p, workload = job_args
    l_dir = os.path.join(
        ROOT_DIR,
        f"logs_{gran}/{MACHINE}_cpu{CPU}_heterogeneous/{mode}/{suite}/{freq}GHz/het_{p}/horizon_{h}/timesteps_{t}"
    )
    return l_dir, os.path.join(l_dir, f"{workload}_{m}.log")

def run_job(job_args):
    gran, suite, freq, h, t, m, mode, p, workload = job_args
    counters = COUNTERS[mode]
    l_dir, log_file = _log_path(job_args)
    os.makedirs(l_dir, exist_ok=True)
    if is_job_successful(log_file):
        return
    dataset_arg = MACHINE
    cmd = [
        "python3", "src/forecasting.py",
        "--benchmark", workload, "--dataset", dataset_arg, "--input_counters"
    ] + counters + [
        "--model", m, "--timesteps", str(t), "--forecast_horizon", str(h),
        "--epochs", str(EPOCHS), "--batch_size", "32", "--neurons", "16",
        "--optimizer", "nadam", "--loss_function", "mae", "--name", f"{workload}_{m}",
        "--heterogeneous_prob", str(p), "--heterogeneous_seed", str(HET_SEED),
        "--heterogeneous_mode", mode,
        "--add_heterogeneity_features",
    ]
    print(f"[*] {workload} | {gran} | {suite} | {freq}GHz | H:{h} | T:{t} | {m} | {mode} | het={p}")
    with open(log_file, "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                       cwd=WORKLOAD_FORECASTING_DIR, env=run_env)

def mode_run_all(jobs):
    print(f"--- FULL SWEEP ({len(jobs)} total jobs) ---")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        executor.map(run_job, jobs)

def mode_rescue(jobs):
    missing = [j for j in jobs if not is_job_successful(_log_path(j)[1])]
    print(f"--- RESCUE: {len(missing)} missing/failed jobs ---")
    if missing:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            executor.map(run_job, missing)
    else:
        print("All jobs complete.")

def mode_condense(jobs):
    for target_gran in GRANULARITIES:
        compiled = []
        for job in [j for j in jobs if j[0] == target_gran]:
            gran, suite, freq, h, t, m, mode, p, workload = job
            _, log_file = _log_path(job)
            record = {
                "workload": workload, "model": m, "frequency": freq,
                "horizon": h, "timesteps": t, "mode": mode, "suite": suite, "het_prob": p,
                "status": "Success" if is_job_successful(log_file) else "Failed/Missing"
            }
            if record["status"] == "Success":
                with open(log_file, 'r', errors='ignore') as f:
                    for key, val in re.findall(
                        r"'([^']+)'\s*:\s*(?:np\.float64\()?([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\)?",
                        f.read()
                    ):
                        record[key] = float(val)
            compiled.append(record)
        out_file = f"heterogeneous_condensed_results_{target_gran}_cpu{CPU}.csv"
        pd.DataFrame(compiled).to_csv(out_file, index=False)
        print(f"Condensed {len(compiled)} jobs → {out_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rescue',   action='store_true')
    parser.add_argument('--condense', action='store_true')
    args = parser.parse_args()
    all_jobs = get_all_jobs()
    if args.rescue:
        mode_rescue(all_jobs)
    elif args.condense:
        mode_condense(all_jobs)
    else:
        mode_run_all(all_jobs)

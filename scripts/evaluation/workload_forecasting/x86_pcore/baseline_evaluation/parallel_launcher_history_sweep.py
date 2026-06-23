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
CPU       = "0"          # P-Core
SUITE     = "spec"
MAX_WORKERS = 80
EPOCHS    = 50

FREQS        = ["1.0", "2.0", "3.0", "4.0"]
HORIZONS     = [1, 2, 4, 8, 16, 32]
TIMESTEPS    = [10]
MODELS       = ["dt", "mlp", "lstm", "transformer"]
GRANULARITIES = ["10M"]

# x86 P-Core (cpu0) counter set
COUNTERS = (
    "cpu_cycles branch_load_misses branch_loads branch_misses branches bus_cycles "
    "cache_misses cache_references dtlb_load_misses dtlb_loads dtlb_store_misses "
    "dtlb_stores fp_arith_scalar_single instructions itlb_load_misses "
    "l1_dcache_load_misses l1_dcache_loads l1_dcache_stores l1_icache_load_misses "
    "llc_loads llc_misses mem_stores ref_cycles"
).split()

SCRIPT_DIR             = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR               = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../../../results/forecasting/"))
WORKLOAD_FORECASTING_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))

run_env = os.environ.copy()
run_env["PYTHONPATH"] = WORKLOAD_FORECASTING_DIR

def get_all_jobs():
    jobs = []
    for gran in GRANULARITIES:
        data_dir = os.path.abspath(
            os.path.join(SCRIPT_DIR, f"../../../../../processed_data_{gran}/{MACHINE}")
        )
        for freq in FREQS:
            pattern = os.path.join(data_dir, "**", f"aligned_{SUITE}_*_{freq}GHz_cpu{CPU}_phase*.csv")
            workloads = sorted(set(
                os.path.basename(f).replace('.csv', '') for f in glob.glob(pattern, recursive=True)
            ))
            for m in MODELS:
                for w in workloads:
                    for h in HORIZONS:
                        for t in TIMESTEPS:
                            jobs.append((gran, freq, h, t, m, w))
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

def run_job(job_args):
    gran, freq, h, t, m, workload = job_args
    l_dir = os.path.join(ROOT_DIR, f"logs_{gran}/{MACHINE}_cpu{CPU}/{freq}GHz/horizon_{h}/timesteps_{t}")
    os.makedirs(l_dir, exist_ok=True)
    log_file = os.path.join(l_dir, f"{workload}_{m}.log")
    if is_job_successful(log_file):
        return
    dataset_arg = MACHINE
    cmd = [
        "python3", "src/forecasting.py",
        "--benchmark", workload, "--dataset", dataset_arg, "--input_counters"
    ] + COUNTERS + [
        "--model", m, "--timesteps", str(t), "--forecast_horizon", str(h),
        "--epochs", str(EPOCHS), "--batch_size", "32", "--neurons", "16",
        "--optimizer", "nadam", "--loss_function", "mae", "--name", f"{workload}_{m}"
    ]
    print(f"[*] {workload} | {gran} | {freq}GHz | H:{h} | T:{t} | {m}")
    with open(log_file, "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                       cwd=WORKLOAD_FORECASTING_DIR, env=run_env)

def mode_run_all(jobs):
    print(f"--- FULL SWEEP ({len(jobs)} total jobs) ---")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        executor.map(run_job, jobs)

def mode_rescue(jobs):
    missing = [j for j in jobs if not is_job_successful(
        os.path.join(ROOT_DIR,
            f"logs_{j[0]}/{MACHINE}_cpu{CPU}/{j[1]}GHz/horizon_{j[2]}/timesteps_{j[3]}/{j[5]}_{j[4]}.log")
    )]
    print(f"--- RESCUE: {len(missing)} missing/failed jobs ---")
    if missing:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            executor.map(run_job, missing)
    else:
        print("All jobs complete.")

def mode_condense(jobs):
    for target_gran in GRANULARITIES:
        compiled = []
        for gran, freq, h, t, m, workload in [j for j in jobs if j[0] == target_gran]:
            log_file = os.path.join(
                ROOT_DIR,
                f"logs_{gran}/{MACHINE}_cpu{CPU}/{freq}GHz/horizon_{h}/timesteps_{t}/{workload}_{m}.log"
            )
            record = {
                "workload": workload, "model": m, "frequency": freq,
                "horizon": h, "timesteps": t,
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
        out_file = f"condensed_results_{target_gran}.csv"
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

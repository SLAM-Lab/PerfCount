import os
import glob
import subprocess
import argparse
import re
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

# =========================================================
# 0. THREADING OPTIMIZATION
# =========================================================
os.environ.update({
    'OMP_NUM_THREADS': '1',
    'MKL_NUM_THREADS': '1',
    'OPENBLAS_NUM_THREADS': '1',
    'BLIS_NUM_THREADS': '1',
    'VECLIB_MAXIMUM_THREADS': '1',
    'NUMEXPR_NUM_THREADS': '1',
    'TF_NUM_INTRAOP_THREADS': '1',
    'TF_NUM_INTEROP_THREADS': '1'
})

# =========================================================
# 1. CONFIGURATION
# =========================================================
MACHINE = "arm_server"
MAX_WORKERS = 80
EPOCHS = 50

FREQS = ["1.0", "2.0", "3.0"]
HORIZONS = [1, 2, 4, 8, 16, 32]
TIMESTEPS = [1, 10]
MODELS = ["dt", "mlp", "lstm", "transformer"]
GRANULARITIES = ["10M"]

COUNTERS = "cpu_cycles branch_misses branches bus_access cache_misses cache_references dtlb_load_misses dtlb_loads instructions itlb-load-misses itlb-loads l1-dcache-load-misses l1-dcache-loads l1_icache_load_misses l1_icache_loads l1d_cache l1d_cache_refill l1d_cache_wb l1i_cache l1i_cache_refill l2d_cache l2d_cache_refill l2d_cache_wb mem_access stalled_cycles_backend stalled_cycles_frontend".split()

# Resolve Base Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../../../results/forecasting/logs_10M_dacapo/"))  
WORKLOAD_FORECASTING_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))

run_env = os.environ.copy()
run_env["PYTHONPATH"] = WORKLOAD_FORECASTING_DIR

# =========================================================
# 2. HELPER FUNCTIONS
# =========================================================
def get_all_jobs():
    """Scans for workloads and builds the master list of all theoretical jobs."""
    jobs = []
    
    for gran in GRANULARITIES:
        # The 10M sweep uses the default 'processed_data' folder.
        # The 100M sweep uses the 'processed_data_100M' folder.
        data_folder = "processed_data" if gran == "10M" else "processed_data_100M"
        data_dir = os.path.abspath(os.path.join(SCRIPT_DIR, f"../../../../../{data_folder}/{MACHINE}"))
        
        for freq in FREQS:
            search_pattern = os.path.join(data_dir, "**", f"aligned_dacapo_*_{freq}GHz_cpu0_phase*.csv")
            csv_files = glob.glob(search_pattern, recursive=True)
            workloads = sorted(list(set([os.path.basename(f).replace('.csv', '') for f in csv_files])))
            
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
        if len(content.strip()) == 0:
            return False
        if "Traceback" in content or "Killed" in content or "Error" in content or "Exception" in content:
            return False
    return True

# =========================================================
# 3. EXECUTION FUNCTION
# =========================================================
def run_job(job_args):
    gran, freq, h, t, m, workload = job_args
    
    # Write only logs, isolating by granularity
    l_dir = os.path.join(ROOT_DIR, f"logs_{gran}/{MACHINE}/{freq}GHz/horizon_{h}/timesteps_{t}")
    os.makedirs(l_dir, exist_ok=True)
    
    log_file = os.path.join(l_dir, f"{workload}_{m}.log")
    
    # Path escape trick: This forces create_dataset.py to look in the right folder 
    # without needing to rewrite your Python source code.
    dataset_arg = MACHINE if gran == "10M" else f"../processed_data_100M/{MACHINE}"
    
    cmd = [
        "python3", "src/forecasting.py",
        "--benchmark", workload,
        "--dataset", dataset_arg,
        "--input_counters"
    ] + COUNTERS + [
        "--model", m,
        "--timesteps", str(t),
        "--forecast_horizon", str(h),
        "--epochs", str(EPOCHS),
        "--batch_size", "32",
        "--neurons", "16",
        "--optimizer", "nadam",
        "--loss_function", "mae",
        "--name", f"{workload}_{m}"
    ]
    
    print(f"[*] Executing: {workload} | {gran} | {freq}GHz | H:{h} | T:{t} | {m}")
    with open(log_file, "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=WORKLOAD_FORECASTING_DIR, env=run_env)

# =========================================================
# 4. MODES
# =========================================================
def mode_run_all(jobs):
    print(f"--- MODE: FULL SWEEP ({len(jobs)} total jobs) ---")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        executor.map(run_job, jobs)

def mode_rescue(jobs):
    missing_jobs = []
    for job in jobs:
        gran, freq, h, t, m, workload = job
        log_file = os.path.join(ROOT_DIR, f"logs_{gran}/{MACHINE}/{freq}GHz/horizon_{h}/timesteps_{t}/{workload}_{m}.log")
        if not is_job_successful(log_file):
            missing_jobs.append(job)
            
    print(f"--- MODE: RESCUE ---")
    if len(missing_jobs) > 0:
        print(f"Launching rescue workers for {len(missing_jobs)} missing/failed jobs...")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            executor.map(run_job, missing_jobs)
    else:
        print("All jobs are successfully completed! Nothing to rescue.")

def mode_condense(jobs):
    print(f"--- MODE: CONDENSE RESULTS ---")
    for target_gran in GRANULARITIES:
        compiled_data = []
        gran_jobs = [j for j in jobs if j[0] == target_gran]
        
        for job in gran_jobs:
            gran, freq, h, t, m, workload = job
            log_file = os.path.join(ROOT_DIR, f"logs_{gran}/{MACHINE}/{freq}GHz/horizon_{h}/timesteps_{t}/{workload}_{m}.log")
            
            job_record = {
                "workload": workload,
                "model": m,
                "frequency": freq,
                "horizon": h,
                "timesteps": t,
                "status": "Success" if is_job_successful(log_file) else "Failed/Missing"
            }
            
            if job_record["status"] == "Success":
                with open(log_file, 'r', errors='ignore') as f:
                    content = f.read()
                    pattern = r"'([^']+)'\s*:\s*(?:np\.float64\()?([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\)?"
                    matches = re.findall(pattern, content)
                    for key, val_str in matches:
                        job_record[key] = float(val_str)
                                    
            compiled_data.append(job_record)
            
        df = pd.DataFrame(compiled_data)
        out_file = f"condensed_results_{target_gran}.csv"
        df.to_csv(out_file, index=False)
        print(f"Successfully condensed {len(compiled_data)} jobs into -> {out_file}")

# =========================================================
# 5. CLI ROUTING
# =========================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Experiment Manager for Workload Forecasting")
    parser.add_argument('--rescue', action='store_true', help='Identify and run ONLY failed/missing jobs')
    parser.add_argument('--condense', action='store_true', help='Compile all logs into CSVs')
    
    args = parser.parse_args()
    all_jobs = get_all_jobs()
    
    # If no flags are provided, default to running the full sweep for both 10M and 100M
    if not args.rescue and not args.condense:
        print("No flags provided. Defaulting to FULL SWEEP for both 10M and 100M...")
        mode_run_all(all_jobs)
    elif args.rescue:
        mode_rescue(all_jobs)
    elif args.condense:
        mode_condense(all_jobs)
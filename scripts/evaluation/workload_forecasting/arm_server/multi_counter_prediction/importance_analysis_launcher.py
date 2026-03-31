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

# Testing all Frequencies and History Lengths
GRANULARITIES = ["10M"]
FREQS = ["1.0", "2.0", "3.0"]
TIMESTEPS = [1, 5, 10] 

# The horizons and models requested for the experiment
HORIZONS = [1, 8, 16, 32]
MODELS = ["dt", "transformer"]

# Master list of counters
COUNTERS = "cpu_cycles branch_misses branches bus_access cache_misses cache_references dtlb_load_misses dtlb_loads instructions itlb-load-misses itlb-loads l1-dcache-load-misses l1-dcache-loads l1_icache_load_misses l1_icache_loads l1d_cache l1d_cache_refill l1d_cache_wb l1i_cache l1i_cache_refill l2d_cache l2d_cache_refill l2d_cache_wb mem_access stalled_cycles_backend stalled_cycles_frontend".split()

# Counters we absolutely CANNOT drop
MANDATORY_COUNTERS = ["cpu_cycles", "instructions"]

# Resolve Base Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../../results/forecasting/feature_importance/"))  
WORKLOAD_FORECASTING_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

run_env = os.environ.copy()
run_env["PYTHONPATH"] = WORKLOAD_FORECASTING_DIR

# =========================================================
# 2. HELPER FUNCTIONS
# =========================================================
def get_all_jobs():
    """Scans for workloads and builds the master list of all ablation jobs."""
    jobs = []
    
    for gran in GRANULARITIES:
        data_folder = "processed_data" if gran == "10M" else "processed_data_100M"
        data_dir = os.path.abspath(os.path.join(SCRIPT_DIR, f"../../../../{data_folder}/{MACHINE}"))
        
        for freq in FREQS:
            search_pattern = os.path.join(data_dir, f"aligned_*_{freq}GHz_phase*.csv")
            csv_files = glob.glob(search_pattern)
            workloads = sorted(list(set([os.path.basename(f).replace('.csv', '') for f in csv_files])))
            
            for m in MODELS:
                for w in workloads:
                    for h in HORIZONS:
                        for t in TIMESTEPS:
                            # 1. Add the Baseline Job (All counters included)
                            jobs.append((gran, freq, h, t, m, w, "Baseline"))
                            
                            # 2. Add Ablation Jobs (Drop one counter at a time)
                            for c in COUNTERS:
                                if c not in MANDATORY_COUNTERS:
                                    jobs.append((gran, freq, h, t, m, w, c))
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
    gran, freq, h, t, m, workload, dropped_counter = job_args
    
    # ADDED TIMESTEP TO PATH TO PREVENT OVERWRITING
    l_dir = os.path.join(ROOT_DIR, f"logs_importance_{gran}/{MACHINE}/{freq}GHz/horizon_{h}/timestep_{t}")
    os.makedirs(l_dir, exist_ok=True)
    
    # Name the log file based on what was dropped
    log_name = f"{workload}_{m}_baseline" if dropped_counter == "Baseline" else f"{workload}_{m}_minus_{dropped_counter}"
    log_file = os.path.join(l_dir, f"{log_name}.log")
    
    # SKIP LOGIC: Don't rerun jobs that already finished successfully
    if is_job_successful(log_file):
        print(f"[SKIP] Already finished: {workload[:15]}... | {freq}GHz | T:{t} H:{h} | Dropped: {dropped_counter}")
        return

    # Build the input counter list for this specific run
    if dropped_counter == "Baseline":
        current_counters = COUNTERS
    else:
        current_counters = [c for c in COUNTERS if c != dropped_counter]
        
    dataset_arg = MACHINE if gran == "10M" else f"../processed_data_100M/{MACHINE}"
    
    cmd = [
        "python3", "src/forecasting.py",
        "--benchmark", workload,
        "--dataset", dataset_arg,
        "--input_counters"
    ] + current_counters + [
        "--model", m,
        "--timesteps", str(t),
        "--forecast_horizon", str(h),
        "--epochs", str(EPOCHS),
        "--batch_size", "32",
        "--neurons", "16",
        "--optimizer", "nadam",
        "--loss_function", "mae",
        "--name", log_name
    ]
    
    print(f"[*] Executing: {workload[:15]}... | {freq}GHz | T:{t} H:{h} | Dropped: {dropped_counter}")
    with open(log_file, "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=WORKLOAD_FORECASTING_DIR, env=run_env)

# =========================================================
# 4. MODES
# =========================================================
def mode_run_all(jobs):
    print(f"--- MODE: ABLATION SWEEP ({len(jobs)} total jobs) ---")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        executor.map(run_job, jobs)

def mode_rescue(jobs):
    missing_jobs = []
    for job in jobs:
        gran, freq, h, t, m, workload, dropped_counter = job
        log_name = f"{workload}_{m}_baseline" if dropped_counter == "Baseline" else f"{workload}_{m}_minus_{dropped_counter}"
        
        # UPDATED PATH HERE AS WELL
        log_file = os.path.join(ROOT_DIR, f"logs_importance_{gran}/{MACHINE}/{freq}GHz/horizon_{h}/timestep_{t}/{log_name}.log")
        
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
    print(f"--- MODE: CONDENSE IMPORTANCE RESULTS ---")
    for target_gran in GRANULARITIES:
        compiled_data = []
        gran_jobs = [j for j in jobs if j[0] == target_gran]
        
        for job in gran_jobs:
            gran, freq, h, t, m, workload, dropped_counter = job
            log_name = f"{workload}_{m}_baseline" if dropped_counter == "Baseline" else f"{workload}_{m}_minus_{dropped_counter}"
            
            # UPDATED PATH HERE
            log_file = os.path.join(ROOT_DIR, f"logs_importance_{gran}/{MACHINE}/{freq}GHz/horizon_{h}/timestep_{t}/{log_name}.log")
            
            job_record = {
                "workload": workload,
                "model": m,
                "freq": freq,         # ADDED FREQ
                "horizon": h,
                "timestep": t,        # ADDED TIMESTEP
                "dropped_counter": dropped_counter,
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
        out_file = f"condensed_importance_{target_gran}.csv"
        df.to_csv(out_file, index=False)
        print(f"Successfully condensed {len(compiled_data)} jobs into -> {out_file}")

# =========================================================
# 5. CLI ROUTING
# =========================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Ablation Feature Importance Manager")
    parser.add_argument('--rescue', action='store_true', help='Identify and run ONLY failed/missing jobs')
    parser.add_argument('--condense', action='store_true', help='Compile all logs into CSVs')
    
    args = parser.parse_args()
    all_jobs = get_all_jobs()
    
    if not args.rescue and not args.condense:
        print("No flags provided. Defaulting to FULL ABLATION SWEEP...")
        mode_run_all(all_jobs)
    elif args.rescue:
        mode_rescue(all_jobs)
    elif args.condense:
        mode_condense(all_jobs)
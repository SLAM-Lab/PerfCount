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
MAX_WORKERS = 100  
EPOCHS = 50

GRANULARITIES = ["10M"]
FREQS = ["1.0", "2.0", "3.0"]
HORIZONS = [1, 8, 16, 32]
TIMESTEPS = [1, 3, 5, 10]
MODELS = ["dt"]

# Dictionary mapping each frequency to its specific Top 5 most important features
TARGET_COUNTERS_BY_FREQ = {
    "1.0": ["branch_misses", "dtlb_load_misses", "bus_access", "dtlb_loads", "cache_misses"],
    "2.0": ["branches", "l1_icache_loads", "l2d_cache_wb", "l1i_cache_refill", "dtlb_loads"],
    "3.0": ["itlb-load-misses", "itlb-loads", "stalled_cycles_frontend", "branches", "l1i_cache_refill"]
}

ALL_COUNTERS = "cpu_cycles branch_misses branches bus_access cache_misses cache_references dtlb_load_misses dtlb_loads instructions itlb-load-misses itlb-loads l1-dcache-load-misses l1-dcache-loads l1_icache_load_misses l1_icache_loads l1d_cache l1d_cache_refill l1d_cache_wb l1i_cache l1i_cache_refill l2d_cache l2d_cache_refill l2d_cache_wb mem_access stalled_cycles_backend stalled_cycles_frontend".split()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../../results/forecasting/"))  
WORKLOAD_FORECASTING_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

run_env = os.environ.copy()
run_env["PYTHONPATH"] = WORKLOAD_FORECASTING_DIR

def get_all_jobs():
    jobs = []
    for gran in GRANULARITIES:
        data_folder = "processed_data" if gran == "10M" else "processed_data_100M"
        data_dir = os.path.abspath(os.path.join(SCRIPT_DIR, f"../../../../{data_folder}/{MACHINE}"))
        
        for freq in FREQS:
            search_pattern = os.path.join(data_dir, f"aligned_*_{freq}GHz_phase*.csv")
            csv_files = glob.glob(search_pattern)
            workloads = sorted(list(set([os.path.basename(f).replace('.csv', '') for f in csv_files])))
            
            target_counters = TARGET_COUNTERS_BY_FREQ.get(freq, [])
            
            for m in MODELS:
                for w in workloads:
                    for h in HORIZONS:
                        for t in TIMESTEPS:
                            for target in target_counters:
                                jobs.append((gran, freq, h, t, m, w, target))
    return jobs

def is_job_successful(log_file):
    if not os.path.exists(log_file):
        return False
    with open(log_file, 'r', errors='ignore') as f:
        content = f.read()
        if len(content.strip()) == 0 or "Traceback" in content or "Error" in content:
            return False
    return True

def run_job(job_args):
    gran, freq, h, t, m, workload, target_counter = job_args
    
    # ADDED TIMESTEP TO PATH HERE
    l_dir = os.path.join(ROOT_DIR, f"logs_dt_freq_multicounter_{gran}/{MACHINE}/{freq}GHz/horizon_{h}/timestep_{t}")
    os.makedirs(l_dir, exist_ok=True)
    
    log_name = f"{workload}_{m}_predicting_{target_counter}"
    log_file = os.path.join(l_dir, f"{log_name}.log")
    
    if is_job_successful(log_file):
        print(f"[SKIP] Already finished: {workload[:15]}... | Freq: {freq} | T:{t} H:{h} | Target: {target_counter}")
        return

    input_list = [target_counter] + [c for c in ALL_COUNTERS if c != target_counter]
    dataset_arg = MACHINE if gran == "10M" else f"../processed_data_100M/{MACHINE}"
    
    cmd = [
        "python3", "src/forecasting.py",
        "--benchmark", workload,
        "--dataset", dataset_arg,
        "--input_counters"
    ] + input_list + [
        "--model", m,
        "--timesteps", str(t),
        "--forecast_horizon", str(h),
        "--epochs", str(EPOCHS),
        "--batch_size", "32",
        "--name", log_name
    ]
    
    print(f"[*] Executing: {workload[:15]}... | Freq: {freq} | T:{t} H:{h} | Target: {target_counter}")
    with open(log_file, "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=WORKLOAD_FORECASTING_DIR, env=run_env)

def mode_run_all(jobs):
    print(f"--- MODE: MULTI-FREQ TARGETED COUNTER SWEEP ({len(jobs)} total jobs) ---")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        executor.map(run_job, jobs)

def mode_condense(jobs):
    for target_gran in GRANULARITIES:
        compiled_data = []
        for job in [j for j in jobs if j[0] == target_gran]:
            gran, freq, h, t, m, workload, target_counter = job
            
            # ADDED TIMESTEP TO CONDENSER PATH HERE
            log_file = os.path.join(ROOT_DIR, f"logs_dt_freq_multicounter_{gran}/{MACHINE}/{freq}GHz/horizon_{h}/timestep_{t}/{workload}_{m}_predicting_{target_counter}.log")
            
            # ADDED 'timestep': t TO THE DICTIONARY
            job_record = {
                "workload": workload, 
                "model": m, 
                "freq": freq, 
                "horizon": h, 
                "timestep": t, 
                "target_predicted": target_counter, 
                "status": "Success" if is_job_successful(log_file) else "Failed"
            }
            
            if job_record["status"] == "Success":
                with open(log_file, 'r', errors='ignore') as f:
                    matches = re.findall(r"'([^']+)'\s*:\s*(?:np\.float64\()?([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\)?", f.read())
                    for key, val_str in matches:
                        job_record[key] = float(val_str)
            compiled_data.append(job_record)
            
        pd.DataFrame(compiled_data).to_csv(f"condensed_dt_freq_multicounter_{target_gran}.csv", index=False)
        print(f"Successfully condensed jobs into condensed_dt_freq_multicounter_{target_gran}.csv")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--condense', action='store_true')
    args = parser.parse_args()
    all_jobs = get_all_jobs()
    
    if args.condense: mode_condense(all_jobs)
    else: mode_run_all(all_jobs)
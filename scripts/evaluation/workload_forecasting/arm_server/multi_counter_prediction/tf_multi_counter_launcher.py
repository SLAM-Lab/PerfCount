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

MACHINE = "arm_server"
MAX_WORKERS = 80  
EPOCHS = 50

# Locked to 10M, 1.0GHz, and the Transformer model
GRANULARITIES = ["10M"]
FREQS = ["1.0"]
HORIZONS = [1, 8, 16, 32]
TIMESTEPS = [1, 5, 10]
MODELS = ["transformer"]

# The Top 5 Transformer counters we want to learn to predict
TARGET_COUNTERS = [
    "l2d_cache",
    "l1_icache_load_misses",
    "l1d_cache_refill",
    "stalled_cycles_frontend",
    "bus_access"
]

ALL_COUNTERS = "cpu_cycles branch_misses branches bus_access cache_misses cache_references dtlb_load_misses dtlb_loads instructions itlb-load-misses itlb-loads l1-dcache-load-misses l1-dcache-loads l1_icache_load_misses l1_icache_loads l1d_cache l1d_cache_refill l1d_cache_wb l1i_cache l1i_cache_refill l2d_cache l2d_cache_refill l2d_cache_wb mem_access stalled_cycles_backend stalled_cycles_frontend".split()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../.."))  
WORKLOAD_FORECASTING_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

run_env = os.environ.copy()
run_env["PYTHONPATH"] = WORKLOAD_FORECASTING_DIR

def get_all_jobs():
    jobs = []
    for gran in GRANULARITIES:
        data_folder = "processed_data" if gran == "10M" else "processed_data_100M"
        data_dir = os.path.abspath(os.path.join(SCRIPT_DIR, f"../../../../{data_folder}/{MACHINE}"))
        for freq in FREQS:
            search_pattern = os.path.join(data_dir, "**", f"aligned_*_{freq}GHz_phase*.csv")
            csv_files = glob.glob(search_pattern, recursive=True)
            workloads = sorted(list(set([os.path.basename(f).replace('.csv', '') for f in csv_files])))
            for m in MODELS:
                for w in workloads:
                    for h in HORIZONS:
                        for t in TIMESTEPS:
                            for target in TARGET_COUNTERS:
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
    
    l_dir = os.path.join(ROOT_DIR, f"logs_tf_multicounter_{gran}/{MACHINE}/{freq}GHz/horizon_{h}")
    os.makedirs(l_dir, exist_ok=True)
    
    log_name = f"{workload}_{m}_predicting_{target_counter}"
    log_file = os.path.join(l_dir, f"{log_name}.log")
    
    # CRITICAL: Force the target counter to index [0] so the pipeline predicts it
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
        "--num_heads", "4",
        "--optimizer", "nadam",
        "--loss_function", "mae",
        "--name", log_name
    ]
    
    print(f"[*] Executing: {workload[:15]}... | {m.upper()} | Target: {target_counter} | H:{h}")
    with open(log_file, "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=WORKLOAD_FORECASTING_DIR, env=run_env)

def mode_run_all(jobs):
    print(f"--- MODE: TF MULTI-COUNTER PREDICTION SWEEP ({len(jobs)} total jobs) ---")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        executor.map(run_job, jobs)

def mode_rescue(jobs):
    missing_jobs = [j for j in jobs if not is_job_successful(os.path.join(ROOT_DIR, f"logs_tf_multicounter_{j[0]}/{MACHINE}/{j[1]}GHz/horizon_{j[2]}/{j[5]}_{j[4]}_predicting_{j[6]}.log"))]
    if len(missing_jobs) > 0:
        print(f"Launching rescue workers for {len(missing_jobs)} missing/failed jobs...")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            executor.map(run_job, missing_jobs)
    else:
        print("All jobs completed!")

def mode_condense(jobs):
    for target_gran in GRANULARITIES:
        compiled_data = []
        for job in [j for j in jobs if j[0] == target_gran]:
            gran, freq, h, t, m, workload, target_counter = job
            log_file = os.path.join(ROOT_DIR, f"logs_tf_multicounter_{gran}/{MACHINE}/{freq}GHz/horizon_{h}/{workload}_{m}_predicting_{target_counter}.log")
            
            job_record = {"workload": workload, "model": m, "horizon": h, "target_predicted": target_counter, "status": "Success" if is_job_successful(log_file) else "Failed"}
            
            if job_record["status"] == "Success":
                with open(log_file, 'r', errors='ignore') as f:
                    matches = re.findall(r"'([^']+)'\s*:\s*(?:np\.float64\()?([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\)?", f.read())
                    for key, val_str in matches:
                        job_record[key] = float(val_str)
            compiled_data.append(job_record)
            
        pd.DataFrame(compiled_data).to_csv(f"condensed_tf_multicounter_{target_gran}.csv", index=False)
        print(f"Successfully condensed jobs into condensed_tf_multicounter_{target_gran}.csv")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rescue', action='store_true')
    parser.add_argument('--condense', action='store_true')
    args = parser.parse_args()
    all_jobs = get_all_jobs()
    
    if args.rescue: mode_rescue(all_jobs)
    elif args.condense: mode_condense(all_jobs)
    else: mode_run_all(all_jobs)
import os
import glob
import subprocess
import argparse
import re
import sys
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations

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
# 1. STATIC CONFIGURATION
# =========================================================
MACHINE = "arm_server"
MAX_WORKERS = 160
EPOCHS = 50

GRANULARITIES = ["10M"]
FREQS = ["1.0", "2.0", "3.0"]
TIMESTEPS = [1, 5, 10]
HORIZONS = [1, 8, 16, 32]

BASE_COUNTERS = ["cpu_cycles", "instructions"]

# Per-model optimized counter presets (used when --extra_counters is not given)
MODEL_EXTRA_COUNTERS = {
    "transformer": [
        "branches",
        "dtlb_load_misses",
        "dtlb_loads",
        "itlb-load-misses",
        "itlb-loads",
        "l1-dcache-loads",
        "l1_icache_load_misses",
        "l1d_cache_wb",
        "l2d_cache",
        "l2d_cache_refill",
        "l2d_cache_wb",
        "stalled_cycles_backend",
        "stalled_cycles_frontend",
    ],
    "dt": [
        "branch_misses",
        "branches",
        "bus_access",
        "cache_misses",
        "cache_references",
        "dtlb_load_misses",
        "dtlb_loads",
        "itlb-load-misses",
        "itlb-loads",
        "l1d_cache_wb",
        "l1i_cache_refill",
        "l2d_cache",
        "l2d_cache_refill",
        "stalled_cycles_backend",
        "stalled_cycles_frontend",
    ],
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../../../results/forecasting/pair_analysis/"))
WORKLOAD_FORECASTING_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))

run_env = os.environ.copy()
run_env["PYTHONPATH"] = WORKLOAD_FORECASTING_DIR


# =========================================================
# 2. HELPER FUNCTIONS
# =========================================================
def get_all_jobs(models, extra_counters):
    """Scans for workloads and builds the master list of all pair jobs."""
    counter_pairs = list(combinations(extra_counters, 2))
    jobs = []

    for gran in GRANULARITIES:
        data_folder = "processed_data" if gran == "10M" else "processed_data_100M"
        data_dir = os.path.abspath(os.path.join(SCRIPT_DIR, f"../../../../../{data_folder}/{MACHINE}"))

        for freq in FREQS:
            search_pattern = os.path.join(data_dir, "**", f"aligned_*_{freq}GHz_phase*.csv")
            csv_files = glob.glob(search_pattern, recursive=True)
            workloads = sorted(set(os.path.basename(f).replace('.csv', '') for f in csv_files))

            for m in models:
                for w in workloads:
                    for h in HORIZONS:
                        for t in TIMESTEPS:
                            jobs.append((gran, freq, h, t, m, w, "Baseline", []))
                            for pair in counter_pairs:
                                pair_name = f"{pair[0]}+{pair[1]}"
                                jobs.append((gran, freq, h, t, m, w, pair_name, list(pair)))
    return jobs


def is_job_successful(log_file):
    if not os.path.exists(log_file):
        return False
    with open(log_file, 'r', errors='ignore') as f:
        content = f.read()
    if not content.strip():
        return False
    if any(tok in content for tok in ("Traceback", "Killed", "Error", "Exception")):
        return False
    return True


def log_path(gran, freq, h, t, m, workload, pair_name):
    log_name = f"{workload}_{m}_baseline" if pair_name == "Baseline" else f"{workload}_{m}_pair_{pair_name}"
    l_dir = os.path.join(ROOT_DIR, f"logs_pairs_{gran}/{MACHINE}/{freq}GHz/horizon_{h}/timestep_{t}")
    return l_dir, log_name, os.path.join(l_dir, f"{log_name}.log")


# =========================================================
# 3. EXECUTION FUNCTION
# =========================================================
def run_job(job_args):
    gran, freq, h, t, m, workload, pair_name, extra_counters_list = job_args
    l_dir, log_name, log_file = log_path(gran, freq, h, t, m, workload, pair_name)
    os.makedirs(l_dir, exist_ok=True)

    if is_job_successful(log_file):
        print(f"[SKIP] Already finished: {workload[:15]}... | {freq}GHz | T:{t} H:{h} | Pair: {pair_name}")
        return

    current_counters = BASE_COUNTERS + extra_counters_list
    dataset_arg = MACHINE if gran == "10M" else f"../processed_data_100M/{MACHINE}"

    cmd = [
        "python3", "src/forecasting.py",
        "--benchmark", workload,
        "--dataset", dataset_arg,
        "--input_counters",
    ] + current_counters + [
        "--model", m,
        "--timesteps", str(t),
        "--forecast_horizon", str(h),
        "--epochs", str(EPOCHS),
        "--batch_size", "32",
        "--neurons", "16",
        "--optimizer", "nadam",
        "--loss_function", "mae",
        "--name", log_name,
    ]

    print(f"[*] Executing: {workload[:15]}... | {freq}GHz | T:{t} H:{h} | Pair: {pair_name}")
    with open(log_file, "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=WORKLOAD_FORECASTING_DIR, env=run_env)


# =========================================================
# 4. MODES
# =========================================================
def mode_run_all(jobs):
    print(f"--- MODE: PAIR SWEEP ({len(jobs)} total jobs) ---")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        executor.map(run_job, jobs)


def mode_rescue(jobs):
    missing_jobs = [
        job for job in jobs
        if not is_job_successful(log_path(*job[:7])[2])
    ]
    print(f"--- MODE: RESCUE ---")
    if missing_jobs:
        print(f"Launching rescue workers for {len(missing_jobs)} missing/failed jobs...")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            executor.map(run_job, missing_jobs)
    else:
        print("All jobs are successfully completed! Nothing to rescue.")


def mode_condense(jobs):
    print(f"--- MODE: CONDENSE PAIR RESULTS ---")
    for target_gran in GRANULARITIES:
        compiled_data = []
        for job in (j for j in jobs if j[0] == target_gran):
            gran, freq, h, t, m, workload, pair_name, _ = job
            _, log_name, log_file = log_path(gran, freq, h, t, m, workload, pair_name)

            job_record = {
                "workload": workload,
                "model": m,
                "freq": freq,
                "horizon": h,
                "timestep": t,
                "pair_name": pair_name,
                "status": "Success" if is_job_successful(log_file) else "Failed/Missing",
            }

            if job_record["status"] == "Success":
                with open(log_file, 'r', errors='ignore') as f:
                    content = f.read()
                pattern = r"'([^']+)'\s*:\s*(?:np\.float64\()?([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\)?"
                for key, val_str in re.findall(pattern, content):
                    job_record[key] = float(val_str)

            compiled_data.append(job_record)

        df = pd.DataFrame(compiled_data)
        out_file = f"condensed_pairs_{target_gran}.csv"
        df.to_csv(out_file, index=False)
        print(f"Successfully condensed {len(compiled_data)} jobs into -> {out_file}")


# =========================================================
# 5. CLI
# =========================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pair Analysis Launcher")
    parser.add_argument('--models', nargs='+', default=None,
                        help=f"Model(s) to run. Known presets: {list(MODEL_EXTRA_COUNTERS)}. "
                             f"Default: transformer")
    parser.add_argument('--extra_counters', nargs='*', default=None,
                        help="Pool of extra counters to pair-combine. "
                             "Defaults to the per-model preset when a single known model is given.")
    parser.add_argument('--rescue', action='store_true',
                        help='Run only failed/missing jobs')
    parser.add_argument('--condense', action='store_true',
                        help='Compile all logs into CSVs')
    args = parser.parse_args()

    models = args.models or ["transformer"]

    if args.extra_counters is not None:
        extra_counters = args.extra_counters
    elif len(models) == 1 and models[0] in MODEL_EXTRA_COUNTERS:
        extra_counters = MODEL_EXTRA_COUNTERS[models[0]]
    else:
        known = list(MODEL_EXTRA_COUNTERS)
        unknown = [m for m in models if m not in MODEL_EXTRA_COUNTERS]
        if unknown or len(models) > 1:
            parser.error(
                f"No counter preset for models {unknown or models}. "
                f"Please supply --extra_counters explicitly."
            )

    print(f"Models: {models}")
    print(f"Extra counters ({len(extra_counters)}): {extra_counters}")

    all_jobs = get_all_jobs(models, extra_counters)

    if args.condense:
        mode_condense(all_jobs)
    elif args.rescue:
        mode_rescue(all_jobs)
    else:
        print("No flags provided. Defaulting to FULL PAIR SWEEP...")
        mode_run_all(all_jobs)

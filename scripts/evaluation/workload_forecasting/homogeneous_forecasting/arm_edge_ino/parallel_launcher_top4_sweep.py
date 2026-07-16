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

MACHINE     = "arm_edge_heterogeneous"
CPU         = "1"           # in-order (little) core
VARIANT     = "top4"
MAX_WORKERS = 70
EPOCHS      = 100
MEM_PER_JOB_GB = 2

FREQS        = ["1.0"]      # only 1GHz data exists for cpu1
HORIZONS     = [1, 5, 10]
TIMESTEPS    = [5, 10]
MODELS       = ["dt", "mlp", "lstm", "transformer"]
GRANULARITIES = ["10M"]

# Top-4 counters derived from arm_server feature importance analysis.
# Same ARM PMU events available on arm_edge.
COUNTERS = "cpu_cycles instructions stalled_cycles_backend l2d_cache_refill".split()

SCRIPT_DIR             = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR               = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../../../results/forecasting/"))
WORKLOAD_FORECASTING_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))

run_env = os.environ.copy()
run_env["PYTHONPATH"] = WORKLOAD_FORECASTING_DIR

def _send_email(subject, body):
    try:
        subprocess.run(
            ["mail", "-s", subject, "-r", "mebarondeau@utexas.edu", "mebarondeau@utexas.edu"],
            input=body.encode(), timeout=30,
        )
    except Exception as e:
        print(f"[EMAIL] Failed to send notification: {e}")

def _available_gb():
    with open('/proc/meminfo') as f:
        for line in f:
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) / (1024 ** 2)
    return float('inf')

def get_all_jobs():
    jobs = []
    for gran in GRANULARITIES:
        data_dir = os.path.abspath(
            os.path.join(SCRIPT_DIR, f"../../../../../processed_data_{gran}/{MACHINE}")
        )
        for freq in FREQS:
            pattern = os.path.join(data_dir, "**", f"aligned_*_{freq}GHz_cpu{CPU}_phase*.csv")
            workloads = sorted(set(
                re.sub(r'_phase\d+$', '', os.path.basename(f).replace('.csv', ''))
                for f in glob.glob(pattern, recursive=True)
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

def _log_dir(gran, freq, h, t):
    return os.path.join(ROOT_DIR, f"logs_{gran}/{MACHINE}_cpu{CPU}_{VARIANT}/{freq}GHz/horizon_{h}/timesteps_{t}")

def _model_dir(gran, freq, h, t):
    return os.path.join(ROOT_DIR, f"models_{gran}/{MACHINE}_cpu{CPU}_{VARIANT}/{freq}GHz/horizon_{h}/timesteps_{t}")

def run_job(job_args):
    gran, freq, h, t, m, workload = job_args
    l_dir = _log_dir(gran, freq, h, t)
    m_dir = _model_dir(gran, freq, h, t)
    os.makedirs(l_dir, exist_ok=True)
    os.makedirs(m_dir, exist_ok=True)

    log_file = os.path.join(l_dir, f"{workload}_{m}.log")
    if is_job_successful(log_file):
        return

    model_path = os.path.join(m_dir, f"{workload}_{m}")
    cmd = [
        "python3", "src/forecasting.py",
        "--benchmark", workload, "--dataset", MACHINE, "--input_counters"
    ] + COUNTERS + [
        "--model", m, "--timesteps", str(t), "--forecast_horizon", str(h),
        "--epochs", str(EPOCHS), "--batch_size", "32", "--neurons", "16",
        "--optimizer", "nadam", "--loss_function", "mae", "--name", f"{workload}_{m}",
        "--save_model_path", model_path,
    ]
    print(f"[*] {workload} | {gran} | {freq}GHz | H:{h} | T:{t} | {m}")
    with open(log_file, "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                       cwd=WORKLOAD_FORECASTING_DIR, env=run_env)

def mode_run_all(jobs):
    print(f"--- FULL SWEEP [{VARIANT}] ({len(jobs)} total jobs) ---")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        executor.map(run_job, jobs)

def mode_rescue(jobs):
    missing = [j for j in jobs if not is_job_successful(
        os.path.join(_log_dir(j[0], j[1], j[2], j[3]), f"{j[5]}_{j[4]}.log")
    )]
    print(f"--- RESCUE [{VARIANT}]: {len(missing)} missing/failed jobs ---")
    if missing:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            executor.map(run_job, missing)
    else:
        print("All jobs complete.")

def mode_condense(jobs):
    for target_gran in GRANULARITIES:
        compiled = []
        for gran, freq, h, t, m, workload in [j for j in jobs if j[0] == target_gran]:
            log_file = os.path.join(_log_dir(gran, freq, h, t), f"{workload}_{m}.log")
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
        out_file = f"condensed_results_{target_gran}_{VARIANT}.csv"
        pd.DataFrame(compiled).to_csv(out_file, index=False)
        print(f"Condensed {len(compiled)} jobs → {out_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=f"Experiment Manager — {MACHINE}_cpu{CPU} {VARIANT} sweep")
    parser.add_argument('--rescue',   action='store_true')
    parser.add_argument('--condense', action='store_true')
    args = parser.parse_args()

    avail = _available_gb()
    mem_cap = max(1, int(avail / MEM_PER_JOB_GB))
    effective_workers = min(MAX_WORKERS, mem_cap)
    if effective_workers < MAX_WORKERS:
        print(f"[MEM] Available: {avail:.1f} GB → capping workers at {effective_workers} (was {MAX_WORKERS})")
    MAX_WORKERS = effective_workers

    all_jobs = get_all_jobs()
    if args.rescue:
        mode_rescue(all_jobs)
        _send_email(
            f"[PerfCount] {MACHINE}_cpu{CPU} {VARIANT} rescue complete on {os.uname().nodename}",
            f"{MACHINE}_cpu{CPU} {VARIANT} rescue finished on {os.uname().nodename}.\nTotal jobs checked: {len(all_jobs)}\n"
        )
    elif args.condense:
        mode_condense(all_jobs)
    else:
        mode_run_all(all_jobs)
        _send_email(
            f"[PerfCount] {MACHINE}_cpu{CPU} {VARIANT} sweep complete on {os.uname().nodename}",
            f"{MACHINE}_cpu{CPU} {VARIANT} full sweep finished on {os.uname().nodename}.\nTotal jobs: {len(all_jobs)}\n"
        )

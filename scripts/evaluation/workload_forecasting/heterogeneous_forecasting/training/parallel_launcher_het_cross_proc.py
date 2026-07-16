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

MACHINE     = "x86_desktop_heterogeneous"
EPOCHS      = 100
HET_SEED    = 42
MEM_PER_JOB_GB = 2

FREQS               = ["1.0", "2.0", "3.0", "4.0"]
HORIZONS            = [1, 5, 10]
TIMESTEPS           = [5]
MODELS              = ["dt", "mlp", "lstm", "transformer"]
GRANULARITIES       = ["10M"]
HET_PROBS           = [0.2, 0.4, 0.6, 0.8, 1.0]

COUNTERS = "ref_cycles cpu_cycles branch_misses instructions".split()

BENCHMARK_NAME_RE = re.compile(
    r"^aligned_(?P<rest>.+)_(?P<freq>[\d.]+)GHz_cpu(?P<cpu>\d+)_phase(?P<phase>\d+)$"
)
SPEC_BENCH_RE = re.compile(r"^spec_(\d+)\.")

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
        print(f"[EMAIL] Failed: {e}")


def _available_gb():
    with open('/proc/meminfo') as f:
        for line in f:
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) / (1024 ** 2)
    return float('inf')


def _suite_of(rest):
    if rest.startswith("dacapo_"):
        return "dacapo"
    m = SPEC_BENCH_RE.match(rest)
    if m:
        return "spec2026" if int(m.group(1)) >= 700 else "spec2017"
    return None


def get_all_jobs(cpu, gran="10M"):
    jobs = []
    data_dir = os.path.abspath(
        os.path.join(SCRIPT_DIR, f"../../../../../processed_data_{gran}/{MACHINE}")
    )
    for freq in FREQS:
        pattern = os.path.join(data_dir, "**", f"aligned_*_{freq}GHz_cpu{cpu}_phase*.csv")
        seen = {}
        for fpath in sorted(glob.glob(pattern, recursive=True)):
            pname = os.path.basename(fpath).replace('.csv', '')
            bm = BENCHMARK_NAME_RE.match(pname)
            if not bm:
                continue
            base_name = f"aligned_{bm.group('rest')}_{freq}GHz_cpu{cpu}"
            if base_name in seen:
                continue
            suite = _suite_of(bm.group('rest'))
            if suite is None:
                continue
            seen[base_name] = suite
        for base_name, suite in seen.items():
            for p in HET_PROBS:
                for m in MODELS:
                    for h in HORIZONS:
                        for t in TIMESTEPS:
                            jobs.append((gran, suite, freq, h, t, m, p, base_name))
    return jobs


def is_job_successful(log_file):
    if not os.path.exists(log_file):
        return False
    with open(log_file, 'r', errors='ignore') as f:
        content = f.read()
    return bool(content.strip()) and not any(
        k in content for k in ("Traceback", "Killed", "Error", "Exception")
    )


def _log_path(job_args, cpu, tag):
    gran, suite, freq, h, t, m, p, workload = job_args
    l_dir = os.path.join(
        ROOT_DIR,
        f"logs_{gran}/{MACHINE}_cpu{cpu}_het_cross_proc_{tag}/{suite}/{freq}GHz/het_{p}/horizon_{h}/timesteps_{t}"
    )
    return l_dir, os.path.join(l_dir, f"{workload}_{m}.log")


def _model_path(job_args, cpu, tag):
    gran, suite, freq, h, t, m, p, workload = job_args
    m_dir = os.path.join(
        ROOT_DIR,
        f"models_{gran}/{MACHINE}_cpu{cpu}_het_cross_proc_{tag}/{suite}/{freq}GHz/het_{p}/horizon_{h}/timesteps_{t}"
    )
    return m_dir, os.path.join(m_dir, f"{workload}_{m}")


def run_job(job_args, cpu, tag, cbm_cross_proc_dir):
    gran, suite, freq, h, t, m, p, workload = job_args
    l_dir, log_file = _log_path(job_args, cpu, tag)
    os.makedirs(l_dir, exist_ok=True)
    if is_job_successful(log_file):
        return
    m_dir, model_path = _model_path(job_args, cpu, tag)
    os.makedirs(m_dir, exist_ok=True)
    early_stop = ["--early_stopping", "--patience", "10"] if m != "dt" else []
    translate_args = ["--cbm_cross_proc_dir", cbm_cross_proc_dir] if cbm_cross_proc_dir else []
    cmd = [
        "python3", "src/forecasting.py",
        "--benchmark", workload, "--dataset", MACHINE, "--input_counters"
    ] + COUNTERS + [
        "--model", m, "--timesteps", str(t), "--forecast_horizon", str(h),
        "--epochs", str(EPOCHS), "--batch_size", "32", "--neurons", "16",
        "--optimizer", "nadam", "--loss_function", "mae", "--name", f"{workload}_{m}",
        "--save_model_path", model_path,
        "--heterogeneous_prob", str(p), "--heterogeneous_seed", str(HET_SEED),
        "--heterogeneous_mode", "cross_proc",
    ] + translate_args + early_stop
    print(f"[*] [{tag}] {workload} | {freq}GHz | H:{h} | T:{t} | {m} | het={p}")
    with open(log_file, "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                       cwd=WORKLOAD_FORECASTING_DIR, env=run_env)


def mode_run_all(jobs, cpu, tag, cbm_cross_proc_dir, max_workers):
    print(f"--- FULL SWEEP [cpu{cpu} het_cross_proc_{tag}] ({len(jobs)} total jobs) ---")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(lambda j: run_job(j, cpu, tag, cbm_cross_proc_dir), jobs)


def mode_rescue(jobs, cpu, tag, cbm_cross_proc_dir, max_workers):
    missing = [j for j in jobs if not is_job_successful(_log_path(j, cpu, tag)[1])]
    print(f"--- RESCUE [cpu{cpu} het_cross_proc_{tag}]: {len(missing)} missing/failed ---")
    if missing:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            executor.map(lambda j: run_job(j, cpu, tag, cbm_cross_proc_dir), missing)
    else:
        print("All jobs complete.")


def mode_condense(jobs, cpu, tag):
    for target_gran in GRANULARITIES:
        compiled = []
        for job in [j for j in jobs if j[0] == target_gran]:
            gran, suite, freq, h, t, m, p, workload = job
            _, log_file = _log_path(job, cpu, tag)
            record = {
                "workload": workload, "model": m, "frequency": freq,
                "horizon": h, "timesteps": t, "suite": suite, "het_prob": p,
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
        # Beside this launcher, not the CWD: plotting_scripts/forecasting/
        # heterogeneous_training/common.py reads them from here by absolute path.
        out_file = os.path.join(
            SCRIPT_DIR, f"het_cross_proc_{tag}_condensed_{target_gran}_cpu{cpu}.csv")
        pd.DataFrame(compiled).to_csv(out_file, index=False)
        print(f"Condensed {len(compiled)} jobs → {out_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=f"Cross-processor het history sweep — {MACHINE}"
    )
    parser.add_argument('--cpu', required=True, choices=['0', '16'],
                        help='CPU core to use: 0=P-core, 16=E-core')
    parser.add_argument('--max_workers', type=int, default=120,
                        help='Maximum parallel jobs (default: 120)')
    parser.add_argument('--no_translate', action='store_true',
                        help='Naive counter swap — no CatBoost ref_cycles translation')
    parser.add_argument('--rescue',   action='store_true')
    parser.add_argument('--condense', action='store_true')
    args = parser.parse_args()

    cpu = args.cpu
    tag = "naive" if args.no_translate else "translated"
    cbm_cross_proc_dir = None if args.no_translate else os.path.abspath(
        os.path.join(SCRIPT_DIR, "../../../../../results/cross_platform/cross_proc/x86_10M")
    )

    avail = _available_gb()
    mem_cap = max(1, int(avail / MEM_PER_JOB_GB))
    max_workers = min(args.max_workers, mem_cap)
    if max_workers < args.max_workers:
        print(f"[MEM] Available: {avail:.1f} GB → capping workers at {max_workers}")

    all_jobs = get_all_jobs(cpu)
    print(f"Mode: {tag} | cpu{cpu} | {len(all_jobs)} total jobs")
    if cbm_cross_proc_dir:
        print(f"CBM cross-proc dir: {cbm_cross_proc_dir}")

    if args.rescue:
        mode_rescue(all_jobs, cpu, tag, cbm_cross_proc_dir, max_workers)
        _send_email(
            f"[PerfCount] cpu{cpu} het_cross_proc_{tag} rescue complete on {os.uname().nodename}",
            f"cpu{cpu} het_cross_proc_{tag} rescue finished.\nJobs checked: {len(all_jobs)}\n"
        )
    elif args.condense:
        mode_condense(all_jobs, cpu, tag)
    else:
        mode_run_all(all_jobs, cpu, tag, cbm_cross_proc_dir, max_workers)
        _send_email(
            f"[PerfCount] cpu{cpu} het_cross_proc_{tag} sweep complete on {os.uname().nodename}",
            f"cpu{cpu} het_cross_proc_{tag} full sweep finished.\nTotal jobs: {len(all_jobs)}\n"
        )

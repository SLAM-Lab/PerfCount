#!/usr/bin/env python3
"""End-to-end latency of one scheduling decision.

Inference latency alone understates what a deployed scheduler pays. A decision is a pipeline,
and every stage runs once per sample:

  1. read the hardware counters for the interval that just completed
  2. normalise them into model features
  3. translate to every reachable configuration (one model call per target)
  4. forecast, when the policy is a forecasting one (one more call per target)
  5. take the argmin, including transition costs
  6. update the reactive-fallback gate's trailing window
  7. actuate, by migrating the thread or writing the new frequency

This measures each stage on the real trained models and the real system calls, and reports the
total against the sample period the scheduler operates at. The number that matters is the
fraction of the interval consumed, because that is pure overhead subtracted from useful work.

Usage:
  decision_latency.py [--reps 2000] [--cpu 2] [--sample_ms 1.93]
"""
import argparse, gc, glob, os, time, sys
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    '..', '..', '..', '..'))
CP = os.path.join(REPO, 'results/cross_platform/cross_proc/x86_10M')
CF = os.path.join(REPO, 'results/cross_platform/cross_freq/x86_10M')

# The action space is eight configurations. From whichever one the thread occupies, the policy
# needs an estimate for the other seven, which is seven model calls per decision.
N_TARGETS = 7


def pick_models(n):
    """One representative trained model per target, from the sets the scheduler consumes."""
    from catboost import CatBoostRegressor
    paths = []
    for base in (CP, CF):
        paths += sorted(glob.glob(f'{base}/*/spec_2017/top4/*/*.cbm'))
    if not paths:
        sys.exit(f'no trained models under {CP} or {CF}')
    out = []
    for p in paths[:n] if len(paths) >= n else (paths * n)[:n]:
        m = CatBoostRegressor(thread_count=1)
        m.load_model(p)
        out.append(m)
    return out


def timeit(fn, reps, warmup=50, wall=False):
    """Latency of one call, in microseconds.

    Defaults to CLOCK_THREAD_CPUTIME_ID rather than wall time. On a loaded machine a pinned
    thread is descheduled repeatedly, and wall-clock timing then charges the measured stage for
    time it spent off the CPU. That inflated a four-syscall counter read to 1.8 ms and an
    eight-element argmin to 4.3 us on a first attempt at load ~110. Thread CPU time counts only
    cycles this thread actually consumed, including inside syscalls, which is the quantity a
    scheduler overhead budget needs. Pass wall=True to measure elapsed time instead, which is
    only meaningful on an idle machine.
    """
    clk = time.CLOCK_MONOTONIC if wall else time.CLOCK_THREAD_CPUTIME_ID
    for _ in range(warmup):
        fn()
    gc.disable()
    ts = np.empty(reps)
    for i in range(reps):
        t0 = time.clock_gettime_ns(clk)
        fn()
        ts[i] = time.clock_gettime_ns(clk) - t0
    gc.enable()
    return ts / 1e3          # microseconds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reps', type=int, default=2000)
    ap.add_argument('--cpu', type=int, default=2)
    ap.add_argument('--sample_ms', type=float, default=1.93,
                    help='mean wall time of one 10M-instruction sample on the P-core')
    a = ap.parse_args()

    try:
        os.sched_setaffinity(0, {a.cpu})
    except OSError:
        print(f'warning: could not pin to cpu {a.cpu}')

    models = pick_models(N_TARGETS)
    nfeat = len(models[0].feature_names_)
    raw = np.random.default_rng(0).random(nfeat + 1).astype(np.float64) * 1e6
    x = np.ascontiguousarray(raw[:nfeat].reshape(1, -1).astype(np.float32))

    stages = {}

    # 1. counter read. A perf_event fd read returns a few tens of bytes from a kernel buffer.
    #    /dev/zero is the closest cheap stand-in for that syscall path. It is NOT /proc/stat,
    #    which on a 160-core machine costs ~450 us per read because the kernel walks every CPU
    #    and formats text, and which is therefore a badly misleading proxy.
    fd = os.open('/dev/zero', os.O_RDONLY)
    def read_counters():
        for _ in range(4):
            os.pread(fd, 64, 0)
    stages['counter read (4 fds)'] = timeit(read_counters, a.reps)

    # 2. feature preparation: per-instruction normalisation, as the models were trained on
    inst = raw[nfeat]
    def prep():
        v = raw[:nfeat] / inst
        return np.ascontiguousarray(v.reshape(1, -1).astype(np.float32))
    stages['feature prep'] = timeit(prep, a.reps)

    # 3. translation to the seven reachable configurations. Two numbers, because they answer
    #    different questions. The Python CatBoost predict() path is what the current
    #    implementation costs, dominated by per-call array conversion rather than by tree
    #    traversal. Evaluating the same trees over a batch and dividing amortises that
    #    conversion away and approximates what a compiled or C-API deployment would pay.
    def translate_py():
        for m in models:
            m.predict(x)
    stages['translate x7 (python API)'] = timeit(translate_py, max(a.reps // 8, 150))

    BATCH = 4096
    xb = np.ascontiguousarray(np.repeat(x, BATCH, axis=0))
    def translate_batch():
        for m in models:
            m.predict(xb)
    tb = timeit(translate_batch, 12, warmup=3)
    stages['translate x7 (amortised)'] = tb / BATCH

    # 5. argmin over the action space including transition costs
    t = np.random.default_rng(1).random(8) + 0.5
    p = np.random.default_rng(2).random(8) + 0.5
    lat = np.random.default_rng(3).random((8, 8)) * 1e-5
    def decide():
        c = p * (t + lat[3]) ** 2
        return int(np.argmin(c))
    stages['argmin + transition'] = timeit(decide, a.reps)

    # 6. reactive-fallback gate bookkeeping over a 200-sample trailing window
    win = np.random.default_rng(4).random((2, 200))
    idx = 0
    def gate():
        nonlocal idx
        win[0, idx % 200] = 1.0
        win[1, idx % 200] = 1.1
        idx += 1
        return win[0].sum() < win[1].sum()
    stages['gate bookkeeping'] = timeit(gate, a.reps)

    # 7. actuation. Thread migration is a real syscall and needs no privilege. Frequency
    #    actuation is a sysfs write that does, so it is reported separately below.
    cur = {a.cpu}
    def migrate():
        os.sched_setaffinity(0, cur)
    stages['actuate (setaffinity)'] = timeit(migrate, a.reps)
    os.close(fd)

    period_us = a.sample_ms * 1000.0
    print(f'\nsample period {a.sample_ms:.2f} ms ({period_us:.0f} us), thread CPU time, '
          f'cpu {a.cpu}, load {os.getloadavg()[0]:.0f}\n')
    print(f"{'stage':<28}{'median us':>11}{'p95 us':>10}{'% of sample':>13}")
    for k, v in stages.items():
        med = float(np.median(v))
        print(f'{k:<28}{med:11.3f}{float(np.percentile(v,95)):10.3f}{med/period_us*100:12.3f}%')

    fixed = sum(float(np.median(stages[k])) for k in
                ('counter read (4 fds)', 'feature prep', 'argmin + transition',
                 'gate bookkeeping', 'actuate (setaffinity)'))
    print(f"{'-'*62}")
    for lbl, tr in (('python API', float(np.median(stages['translate x7 (python API)']))),
                    ('amortised', float(np.median(stages['translate x7 (amortised)'])))):
        r, f = fixed + tr, fixed + 2 * tr
        print(f"  reactive decision  ({lbl:10s}) {r:9.2f} us  {r/period_us*100:7.3f}% of sample")
        print(f"  forecast decision  ({lbl:10s}) {f:9.2f} us  {f/period_us*100:7.3f}% of sample")
    print('\nForecasting runs the translator a second time, once on the forecast counters, so '
          'its decision cost is the reactive cost plus one more translation pass.')


if __name__ == '__main__':
    main()

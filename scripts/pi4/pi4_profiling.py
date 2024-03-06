#!/usr/bin/env python3


if __name__ == "__main__":
    benchmarks = [
                600, 602, 605, 620, 623, 625, 631, 641, 648, 657,
                603, 607, 619, 621, 627, 628, 638, 644, 649, 654,
                500, 502, 505, 520, 523, 525, 531, 541, 548, 557,
                503, 507, 508, 510, 511, 519, 521, 526, 527, 538, 544, 549, 554]
    perf_events = [
                "inst_retired,cpu_cycles,br_mis_pred,br_pred",
                "l1d_cache,l1d_cache_rd,l1d_cache_wr",
                "l1i_cache,l2d_cache,l2d_cache_rd,l2d_cache_wr",
                "mem_acces,mem_access_rd,mem_access_wr,vfp_spec",
    ]

    sample_period = [10]
    cpus = [0]
    freqs = ['1.80GHz', '1.00GHz']

    for cpu in cpus:
        for freq in freqs:
            print('cpupower frequency-set --max ' + str(freq))
            print('cpupower frequency-set -g performance')
            for epoch in sample_period:
                for benchmark in benchmarks:
                    for events in perf_events:
                        command  = "taskset --cpu-list " + str(cpu)
                        command += ' perf stat -I ' + str(epoch)
                        command += ' -e ' + str(events)
                        command += ' -o cpu_' + str(cpu) + '_freq_' + str(freq) + '_epoch_' + str(epoch) + '_' + str(benchmark) + '_' + str(events)
                        command += ' -x,'
                        command += " runcpu --config=matthew-1cpu " + str(benchmark)
                        print(command)

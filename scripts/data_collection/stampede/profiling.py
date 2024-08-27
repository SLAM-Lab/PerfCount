#!/usr/bin/env python3

import os
import time

if __name__ == "__main__":
    benchmarks = [
                500, 502, 505, 520, 523, 525, 531, 541, 548, 557,
                503, 507, 508, 510, 511, 519, 521, 526, 527, 538, 544, 549, 554]

    perf_events = [
                "instructions,cpu-cycles,br_misp_retired.all_branches,br_inst_retired.all_branches",
                "instructions,L1-dcache-loads-misses,L1-dcache-loads,L1-dcache-stores",
                "instructions,branch-load-misses,branch-loads,dTLB-stores",
                "instructions,dTLB-load-misses,dTLB-loads,dTLB-store-misses",
                "instructions,iTLB-load-misses,l2_request.all,l2_request.miss",
                "instructions,mem_store_retired.l2_hit,LLC-load-misses",
                "instructions,LLC-loads,LLC-store-misses,LLC-stores",
                "instructions,mem-loads,mem-stores,context-switches"
                ]

    sample_period = [10]
    cpus = [0]
    freqs = ['3.00GHz']

    for cpu in cpus:
        for freq in freqs:
            print('cpupower frequency-set --max ' + str(freq))
            print('cpupower frequency-set -g performance')
            for epoch in sample_period:
                for benchmark in benchmarks:
                    count = 0
                    for events in perf_events:
                        command  = "taskset --cpu-list " + str(cpu)
                        command += ' perf stat -I ' + str(epoch)
                        command += ' -e ' + str(events)
                        command += ' -o cpu_' + str(cpu) + '_freq_' + str(freq) + '_epoch_' + str(epoch) + '_' + str(benchmark) + '_' + str(count)
                        command += ' -x,'
                        command += " runcpu --config=matthew-1cpu " + str(benchmark)
                        print(command)
                        count += 1


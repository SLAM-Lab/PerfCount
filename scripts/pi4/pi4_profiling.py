#!/usr/bin/env python3

import os
import time

if __name__ == "__main__":
    benchmarks = [600, 602, 605, 620, 623, 625, 631, 641, 648, 657]
    perf_events = ["inst_retired,cpu_cycles,l1d_cache_rd,l1i_cache",
        "br_mis_pred,l2d_cache_rd,mem_access_rd,vfp_spec"
    ]

    sample_period = [1, 2, 4, 10]
    cpus = [0]
    freqs = ['1.80GHz', '1.00GHz']


    time.sleep(30)

    for cpu in cpus:
        for freq in freqs:
            for epoch in sample_period:
                for benchmark in benchmarks:
                    for events in perf_events:
                        command  = "taskset --cpu-list " + str(cpu)
                        command += ' perf stat -I ' + str(epoch)
                        command += ' -e ' + str(events)
                        command += ' -o cpu_' + str(cpu) + '_freq_' + str(freq) + '_epoch_' + str(epoch) + '_' + str(benchmark) + '_' + str(events)
                        command += ' -x,'
                        command += " runcpu --config=matthew-1cpu " + str(benchmark)
                        os.system('cpupower frequency-set --max ' + str(freq))
                        os.system('cpupower frequency-set -g performance')
                        os.system(command)

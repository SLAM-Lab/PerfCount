#!/usr/bin/env python3

import os

MAX_PERF_EVENTS = 4

if __name__ == "__main__":
    benchmarks = [600, 602, 605, 620, 623, 625, 631, 641, 648, 657]
    perf_events = ["inst_retired,cpu_cycles,l1d_cache_rd,l1i_cache",
        "br_mis_pred,l2d_cache_rd,mem_access_rd,vfp_spec"
    ]

    sample_period = [1, 2, 4, 10]
    cpus = [0]


    for cpu in cpus:
        for epoch in sample_period:
            for benchmark in benchmarks:
                for events in perf_events:
                    command  = "taskset --cpu-list " + str(cpu)
                    command += ' perf stat -I ' + str(epoch)
                    command += ' -e ' + str(events)
                    command += ' -o cpu_' + str(cpu) + '_epoch_' + str(epoch) + '_' + str(benchmark) + '_' + str(events)
                    command += ' -x,'
                    command += " runcpu --config=matthew-1cpu " + str(benchmark)
                    os.system(command)

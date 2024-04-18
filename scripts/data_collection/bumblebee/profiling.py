#!/usr/bin/env python3

import os
import time

if __name__ == "__main__":
    benchmarks = [
#                600, 602, 605, 620, 623, 625, 631, 641, 648, 657, 
#                603, 607, 619, 621, 627, 628, 638, 644, 649, 654,
                500, 502, 505, 520, 523, 525, 531, 541, 548, 557,
                503, 507, 508, 510, 511, 519, 521, 526, 527, 538, 544, 549, 554]
    perf_events_cpu = ["cpu_core/instructions/,cpu_core/cpu-cycles/"]#,cpu_core/topdown-be-bound/,cpu_core/topdown-fe-bound/"]
    perf_events_atom = ['cpu_atom/instructions/,cpu_atom/cpu-cycles/']#,cpu_atom/topdown-be-bound/,cpu_atom/topdown-fe-bound/']
# br_pred,br_mis_pred",
#        "L1-dcache-load-misses,L1-dcache-loads,L1-icache-load-misses,L1-icache-loads",
#        "l2_cache,l2_cache_rd,l3d_cache,l3d_cache_rd",
#        "vfp_spec,mem_access,mem_access_rd",
#        "stall_backend,stall_frontend"
#    ]

    sample_period = [10]
    cpus = [0, 8]
    atom_cpus = [16]
    freqs = ['2.00GHz']
    for cpu in cpus:
        for freq in freqs:
           for epoch in sample_period:
                for benchmark in benchmarks:
                    for events in perf_events_cpu:
                        command  = "taskset --cpu-list " + str(cpu)
                        command += ' perf stat -I ' + str(epoch)
                        command += ' -e ' + str(events)
                        command += ' -o cpu_' + str(cpu) + '_freq_' + str(freq) + '_epoch_' + str(epoch) + '_' + str(benchmark)# + '_' + str(events)
                        command += ' -x,'
                        command += " runcpu --config=matthew-1cpu " + str(benchmark)
                        print(command)

    for cpu in atom_cpus:
        for freq in freqs:
           for epoch in sample_period:
                for benchmark in benchmarks:
                    for events in perf_events_atom:
                        command  = "taskset --cpu-list " + str(cpu)
                        command += ' perf stat -I ' + str(epoch)
                        command += ' -e ' + str(events)
                        command += ' -o cpu_' + str(cpu) + '_freq_' + str(freq) + '_epoch_' + str(epoch) + '_' + str(benchmark)# + '_' + str(events)
                        command += ' -x,'
                        command += " runcpu --config=matthew-1cpu " + str(benchmark)
                        print(command)


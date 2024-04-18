#!/usr/bin/env python3

import os
import time

if __name__ == "__main__":
    benchmarks = [
#                623, 625, 631, 641, 648, 657, 
#                603, 607, 619, 621, 627, 628, 638, 644, 649, 654,
                500, 502, 505, 520, 523, 525, 531, 541, 548, 557,
                503, 507, 508, 510, 511, 519, 521, 526, 527, 538, 544, 549, 554]
    perf_events = [
                "instructions,cpu-cycles,stall_backend,stall_frontend",
                "instructions,br_mis_pred,br_pred,branch-loads",
                "instructions,br_mis_pred_retired,br_retired",
                "instructions,inst_spec,inst_retired",
                "instructions,ld_spec,st_spec,vfp_spec",
                "instructions,l1d_cache,l1d_cache_rd,l1d_cache_wr",
                "instructions,l1d_tlb,l1d_tlb_rd,l1d_tlb_refill"
                "instructions,l1i_cache,l1i_cache_refill",
                "instructions,l1i_tlb,l1i_tlb_refill"
                "instructions,l2d_cache,l2d_cache_rd,l2d_cache_wr",
                "instructions,l2d_tlb,l2d_tlb_rd,l2d_tlb_wr",
                "instructions,l3d_cache,l3d_cache_rd",
                "instructions,mem_access,mem_access_rd,mem_access_wr" 
                ]
#br_pred,br_mis_pred",
#        "L1-dcache-load-misses,L1-dcache-loads,L1-icache-load-misses,L1-icache-loads",
#        "l2d_cache,l2d_cache_rd,l3d_cache,l3d_cache_rd",
#        "vfp_spec,mem_access,mem_access_rd",
#        "stall_backend,stall_frontend"
#    ]

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


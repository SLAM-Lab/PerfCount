#!/usr/bin/env python3

import os
import time

if __name__ == "__main__":
    benchmarks = [
                500, 502, 505, 520, 523, 525, 531, 541, 548, 557,
                503, 507, 508, 510, 511, 519, 521, 527, 538, 544, 549, 554
    ]
    perf_events = [
                "cpu_core/instructions/,cpu_core/cpu-cycles/,cpu_core/br_inst_retired.all_branches/,cpu_core/br_misp_retired.all_branches/",
                "cpu_core/instructions/,cpu_core/L1-dcache-load-misses/,cpu_core/L1-dcache-loads/,cpu_core/L1-dcache-stores/",
                "cpu_core/instructions/,cpu_core/L1-icache-load-misses/,cpu_core/LLC-load-misses/,cpu_core/LLC-loads/",
                "cpu_core/instructions/,cpu_core/LLC-store-misses/,cpu_core/LLC-stores/,cpu_core/fp_arith_inst_retired.scalar_single/",
                "cpu_core/instructions/,cpu_core/dTLB-load-misses/,cpu_core/dTLB-loads/,cpu_core/dTLB-store-misses/",
                "cpu_core/instructions/,cpu_core/dTLB-stores/,cpu_core/iTLB-load-misses/,cpu_core/bus-cycles/",
                "cpu_core/instructions/,cpu_core/mem_load_retired.l1_hit/,cpu_core/mem_load_retired.l1_miss/,cpu_core/mem_load_retired.l2_hit/",
                "cpu_core/instructions/,cpu_core/mem_load_retired.l2_miss/,cpu_core/mem_load_retired.l3_hit/,cpu_core/mem_load_retired.l3_miss/"
                ]

    atom_events = [
                "cpu_atom/instructions/,cpu_atom/cpu-cycles/,cpu_atom/br_inst_retired.all_branches/,cpu_atom/br_misp_retired.all_branches/",
                "cpu_atom/instructions/,cpu_atom/cache-misses/,cpu_atom/L1-dcache-loads/,cpu_atom/L1-dcache-stores/",
                "cpu_atom/instructions/,cpu_atom/L1-icache-load-misses/,cpu_atom/LLC-load-misses/,cpu_atom/LLC-loads/",
                "cpu_atom/instructions/,cpu_atom/LLC-store-misses/,cpu_atom/LLC-stores/,cpu_atom/fp_arith_inst_retired.scalar_single/",
                "cpu_atom/instructions/,cpu_atom/dTLB-load-misses/,cpu_atom/dTLB-loads/,cpu_atom/dTLB-store-misses/",
                "cpu_atom/instructions/,cpu_atom/dTLB-stores/,cpu_atom/iTLB-load-misses/,cpu_atom/bus-cycles/",
                "cpu_atom/instructions/,cpu_atom/mem_load_retired.l1_hit/,cpu_atom/mem_load_retired.l1_miss/,cpu_atom/mem_load_retired.l2_hit/",
                "cpu_atom/instructions/,cpu_atom/mem_load_retired.l2_miss/,cpu_atom/mem_load_retired.l3_hit/,cpu_atom/mem_load_retired.l3_miss/"
                ]



    sample_period = [10]
    cpus = [0]
    atom_cpus = [16]
    freqs = ['1.50GHz']

    for cpu in cpus:
        for freq in freqs:
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

    for cpu in atom_cpus:
        for freq in freqs:
            for epoch in sample_period:
                for benchmark in benchmarks:
                    count = 0
                    for events in atom_events:
                        command  = "taskset --cpu-list " + str(cpu)
                        command += ' perf stat -I ' + str(epoch)
                        command += ' -e ' + str(events)
                        command += ' -o cpu_' + str(cpu) + '_freq_' + str(freq) + '_epoch_' + str(epoch) + '_' + str(benchmark) + '_' + str(count)
                        command += ' -x,'
                        command += " runcpu --config=matthew-1cpu " + str(benchmark)
                        print(command)
                        count += 1


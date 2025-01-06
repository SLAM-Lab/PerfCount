#!/usr/bin/env python3

import os
import time

if __name__ == "__main__":
    benchmarks = [
                500, 502, 505, 520, 523, 525, 531, 541, 548, 557,
                503, 507, 508, 510, 511, 519, 521, 527, 538, 544, 549, 554]
    perf_events = [
                "instructions,cpu-cycles,br_inst_retired.all_branches_pebs,br_misp_retired.all_branches_pebs",
                "instructions,L1-dcache-load-misses,L1-dcache-loads,L1-dcache-stores",
                "instructions,L1-icache-load-misses,LLC-load-misses,LLC-loads",
                "instructions,LLC-store-misses,LLC-stores,fp_assist.any",
                "instructions,dTLB-load-misses,dTLB-loads,dTLB-store-misses",
                "instructions,dTLB-stores,iTLB-load-misses,iTLB-loads",
                "instructions,l1d.replacement,l2_lines_in.all,l2_rqsts.all_pf",
                "instructions,l2_rqsts.all_code_rd,l2_rqsts.code_rd_hit,l2_rqsts.code_rd_miss",
                "instructions,l2_rqsts.all_demand_miss,l2_rqsts.all_demand_references,l2_rqsts.miss",
                "instructions,mem_inst_retired.all_loads,mem_inst_retired.all_stores,mem_load_retired.fb_hit",
                "instructions,mem_load_retired.l1_hit,mem_load_retired.l1_miss,mem_load_retired.l2_hit",
                "instructions,mem_load_retired.l2_miss,mem_load_retired.l3_hit,mem_load_retired.l3_miss",
                "instructions,fp_arith_inst_retired.128b_packed_double,fp_arith_inst_retired.128b_packed_single,fp_arith_inst_retired.256b_packed_double",
                "instructions,fp_arith_inst_retired.256b_packed_single,fp_arith_inst_retired.scalar_double,fp_arith_inst_retired.scalar_single",
                "instructions,frontend_retired.itlb_miss,frontend_retired.l1i_miss,frontend_retired.l2_miss",
                "instructions,cycle_activity.stalls_total,cycle_activity.stalls_mem_any,cycle_activity.cycles_mem_any",
                "instructions,uops_executed.core,uops_issued.any,uops_retired.stall_cycles"
                ]

    sample_period = [10]
    cpus = [0]
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


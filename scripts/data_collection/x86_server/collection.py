#!/usr/bin/env python3

import os
import time

if __name__ == "__main__":
    # SPEC 2017 benchmarks
    spec_benchmarks = [
         500, 502, 505, 520, 523, 525, 531, 541, 548, 557,
         503, 507, 508, 510, 511, 519, 521, 526, 527, 538, 544, 549, 554
    ]
    
    # DaCapo benchmarks
    dacapo_benchmarks = [
        'avrora', 'batik', 'biojava', 'cassandra', 'eclipse', 'fop', 'graphchi', 'h2', 'h2o', 'jme',
        'jython', 'kafka', 'luindex', 'lusearch', 'pmd', 'spring', 'sunflow', 'tomcat', 'tradebeans', 'tradesoap', 'xalan', 'zxing'
    ]
      
    # X86 SERVER EVENT GROUPS - 4 COUNTERS PER GROUP (OPTIMIZED FOR ICE LAKE)
    # 16 groups, 64 unique events
    
    # WORKING GROUPS (perfect 10M instruction chunks):
    
    # Group 0: Cache Performance
    x86_group0 = "\"{instructions,cache-misses,cache-references,bus-cycles}:S\""
    
    # Group 1: Reference Cycles and TLB Events
    x86_group1 = "\"{instructions,ref-cycles,dtlb_load_misses.walk_completed,itlb_misses.walk_completed}:S\""
    
    # Group 2: L1 Data Cache and TLB Events
    x86_group2 = "\"{instructions,l1d_pend_miss.fb_full,l1d_pend_miss.l2_stall,dtlb_load_misses.stlb_hit}:S\""
    
    # Group 3: L2 Cache Demand Events
    x86_group3 = "\"{instructions,l2_lines_in.all,l2_rqsts.demand_data_rd_hit,l2_rqsts.demand_data_rd_miss}:S\""
    
    # Group 4: L2 Cache All Events
    x86_group4 = "\"{instructions,l2_rqsts.all_demand_data_rd,l2_rqsts.all_demand_miss,l2_lines_out.non_silent}:S\""
    
    # Group 5: TLB Walk Active Events
    x86_group5 = "\"{instructions,dtlb_load_misses.walk_active,dtlb_store_misses.walk_active,itlb_misses.walk_active}:S\""
    
    # Group 6: L2 Cache Code Events
    x86_group6 = "\"{instructions,l2_rqsts.code_rd_hit,l2_rqsts.code_rd_miss,l2_rqsts.all_code_rd}:S\""
    
    # Group 7: L1 Data Cache Events
    x86_group7 = "\"{instructions,l1d.replacement,l1d_pend_miss.pending_cycles,l1d_pend_miss.pending}:S\""
    
    # Group 8: Data TLB Events
    x86_group8 = "\"{instructions,dtlb_load_misses.walk_pending,dtlb_store_misses.walk_pending}:S\""
    
    # Group 9: Instruction TLB Events
    x86_group9 = "\"{instructions,itlb_misses.stlb_hit,itlb_misses.walk_pending}:S\""
    
    # Group 10: AVX Events
    x86_group10 = "\"{instructions,fp_arith_inst_retired.256b_packed_single,fp_arith_inst_retired.256b_packed_double,fp_arith_inst_retired.512b_packed_single}:S\""
    
    # Group 11: Data TLB Store Events
    x86_group11 = "\"{instructions,dtlb_store_misses.stlb_hit,dtlb_store_misses.walk_completed}:S\""
    
    # ALTERNATIVE GROUPS (with working alternative counters):
    
    # Group 12: Basic Performance Counters (with alternative)
    x86_group12 = "\"{instructions,cpu-cycles,branches,branch-misses}:S\""
    
    # Group 13: Memory Load Events (with alternative)
    x86_group13 = "\"{instructions,mem_load_retired.fb_hit,mem_load_retired.l1_miss,mem_load_retired.l2_hit,mem_load_retired.l3_hit}:S\""
    
    # Group 14: L1 Instruction Cache Events (with alternative)
    x86_group14 = "\"{instructions,L1-icache-load-misses,icache_64b.iftag_miss,icache_64b.iftag_stall}:S\""
    
    # Group 15: Floating Point Events (with alternative)
    x86_group15 = "\"{instructions,fp_arith_inst_retired.scalar_single,fp_arith_inst_retired.128b_packed_double,fp_arith_inst_retired.128b_packed_single}:S\""

    # Combine all groups for easy iteration
    # First 12 groups: WORKING (perfect 10M instruction chunks)
    # Last 4 groups: ALTERNATIVE (with working alternative counters)
    x86_groups = [
        x86_group0, x86_group1, x86_group2, x86_group3,
        x86_group4, x86_group5, x86_group6, x86_group7,
        x86_group8, x86_group9, x86_group10, x86_group11,
        x86_group12, x86_group13, x86_group14, x86_group15
    ]

    # Intel Xeon Platinum 8380 server configuration (Ice Lake)
    cpus = [0]  # Multiple CPU cores to test
    freq = '2.3GHz'  # Base frequency for Intel Xeon Platinum 8380
    
    # Granularity: 10M instructions
    granularities = [10000000]
    
    # DaCapo directory
    dacapo_dir = '../../../benchmarks/dacapo/'
    
    # SPEC directory  
    spec_dir = '../../../benchmarks/spec'

    # Enhanced perf configuration for Intel Xeon Platinum 8380 (Ice Lake):
    # - --no-buffering: Disables internal buffering for immediate sample writing
    # - taskset --cpu-list: Pins process to specific CPU core for consistent execution
    # - Ice Lake-specific events optimized for Ice Lake microarchitecture
    
    # Single run configuration for initial testing
    redundancy_runs = [1]
    
    for redundancy in redundancy_runs:
        for cpu in cpus:
            for benchmark in spec_benchmarks:
                for granularity in granularities:
                    count = 0
                    for events in x86_groups:
                        command  = "perf record -a -C " + str(cpu)
                        command += " -e " + str(events)
                        command += " -c " + str(granularity)
                        command += " --no-buffering"
                        command += " -o cpu_" + str(cpu) + "_" + str(freq) + "_spec_" + str(benchmark) + "_" + str(granularity) + "_" + str(count) + "_" + str(redundancy) + ".out"
                        command += " taskset --cpu-list " + str(cpu)
                        command += " bash -c 'cd " + spec_dir + " && source shrc && runcpu -c base --config=matthew-1cpu-x86 " + str(benchmark) + "'"
                        print(command)
                        count += 1
    
    # DaCapo benchmarks with enhanced perf configuration for x86 Ice Lake
    for cpu in cpus:
        for benchmark in dacapo_benchmarks:
            for granularity in granularities:
                count = 0
                for events in x86_groups:
                    command  = "perf record -a -C " + str(cpu)
                    command += " -e " + str(events)
                    command += " -c " + str(granularity)
                    command += " --no-buffering"
                    command += " -o cpu_" + str(cpu) + "_" + str(freq) + "_dacapo_" + str(benchmark) + "_" + str(granularity) + "_" + str(count) + ".out"
                    command += " taskset --cpu-list " + str(cpu)
                    command += " java -XX:+UseSerialGC -Xint -XX:ParallelGCThreads=1 -XX:CICompilerCount=1 -XX:-BackgroundCompilation -XX:-TieredCompilation -Xms2g -Xmx2g -XX:-UseAdaptiveSizePolicy -jar " + dacapo_dir + "dacapo-23.11-MR2-chopin.jar -t 1 -n 1 " + str(benchmark)
                    # print(command)
                    count += 1

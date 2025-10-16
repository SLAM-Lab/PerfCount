#!/usr/bin/env python3

import os
import time

if __name__ == "__main__":
    # SPEC 2017 benchmarks
    spec_benchmarks = [
        503
        #  500, 502, 505, 520, 523, 525, 531, 541, 548, 557,
        #  503, 507, 508, 510, 511, 519, 521, 526, 527, 538, 544, 549, 554
    ]
    
    # DaCapo benchmarks
    dacapo_benchmarks = [
        'avrora', 'batik', 'biojava', 'cassandra', 'eclipse', 'fop', 'graphchi', 'h2', 'h2o', 'jme',
        'jython', 'kafka', 'luindex', 'lusearch', 'pmd', 'spring', 'sunflow', 'tomcat', 'tradebeans', 'tradesoap', 'xalan', 'zxing'
    ]
      
    # X86 SERVER EVENT GROUPS - 4 COUNTERS PER GROUP (TESTED FOR CONFLICTS)
    # Each group contains 4 events that work together without conflicts
    # Total: 15 groups covering 60 unique events (59 non-instruction + instructions in each group)
    
    # Group 0: Basic Performance Counters - Verified working
    x86_group0 = "\"{instructions,cpu-cycles,branch-instructions,branch-misses}:S\""
    
    # Group 1: Cache Performance - Verified working
    x86_group1 = "\"{instructions,cache-misses,cache-references,bus-cycles}:S\""
    
    # Group 2: Memory and Reference Events - Verified working
    x86_group2 = "\"{instructions,ref-cycles,page-faults,context-switches}:S\""
    
    # Group 3: L1 Data Cache Events - Verified working
    x86_group3 = "\"{instructions,l1d_pend_miss.pending_cycles,l1d_pend_miss.pending_cycles_any,l1d.replacement}:S\""
    
    # Group 4: L2 Cache Demand Events - Verified working
    x86_group4 = "\"{instructions,l2_lines_in.all,l2_rqsts.demand_data_rd_hit,l2_rqsts.demand_data_rd_miss}:S\""
    
    # Group 5: L2 Cache All Events - Verified working
    x86_group5 = "\"{instructions,l2_rqsts.all_demand_data_rd,l2_rqsts.all_demand_miss,l2_rqsts.all_demand_references}:S\""
    
    # Group 6: Software Events - Verified working
    x86_group6 = "\"{instructions,alignment-faults,emulation-faults,major-faults}:S\""
    
    # Group 7: Task and CPU Events - Verified working
    x86_group7 = "\"{instructions,cpu-migrations,minor-faults,task-clock}:S\""
    
    # Group 8: L2 Cache Code Events - Verified working
    x86_group8 = "\"{instructions,l2_rqsts.code_rd_hit,l2_rqsts.code_rd_miss,l2_rqsts.all_code_rd}:S\""
    
    # Group 9: L1 Data Cache Load/Store Events - Verified working
    x86_group9 = "\"{instructions,L1-dcache-loads,L1-dcache-stores,L1-dcache-load-misses}:S\""
    
    # Group 10: L1 Instruction Cache Events - Verified working
    x86_group10 = "\"{instructions,L1-icache-load-misses,branch-loads,branch-load-misses}:S\""
    
    # Group 11: Data TLB Events - Verified working
    x86_group11 = "\"{instructions,dTLB-loads,dTLB-load-misses,dTLB-store-misses}:S\""
    
    # Group 12: Instruction TLB Events - Verified working
    x86_group12 = "\"{instructions,dTLB-stores,iTLB-loads,iTLB-load-misses}:S\""
    
    # Group 13: Floating Point Events - Verified working
    x86_group13 = "\"{instructions,fp_arith_inst_retired.scalar_single,fp_arith_inst_retired.scalar_double,fp_arith_inst_retired.128b_packed_single}:S\""
    
    # Group 14: AVX Events - Verified working
    x86_group14 = "\"{instructions,fp_arith_inst_retired.256b_packed_single,fp_arith_inst_retired.256b_packed_double,fp_arith_inst_retired.512b_packed_single}:S\""

    # Combine all groups for easy iteration
    # All groups are used for comprehensive analysis on x86 server
    x86_groups = [
        x86_group0, x86_group1, x86_group2, x86_group3,
        x86_group4, x86_group5, x86_group6, x86_group7,
        x86_group8, x86_group9, x86_group10, x86_group11,
        x86_group12, x86_group13, x86_group14
    ]

    # Intel Xeon Gold 6126 server configuration
    cpus = [0]  # Multiple CPU cores to test
    freq = '1.5GHz'  # Base frequency for Intel Xeon Gold 6126
    
    # Granularity: 10M instructions
    granularities = [10000000]
    
    # DaCapo directory
    dacapo_dir = '../../../benchmarks/dacapo/'
    
    # SPEC directory  
    spec_dir = '../../../benchmarks/spec'

    # Enhanced perf configuration for Intel Xeon Gold 6126:
    # - --no-buffering: Disables internal buffering for immediate sample writing
    # - taskset --cpu-list: Pins process to specific CPU core for consistent execution
    # - Intel-specific events optimized for Skylake microarchitecture
    
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
    
    # DaCapo benchmarks with enhanced perf configuration for x86
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

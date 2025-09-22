#!/usr/bin/env python3

import os
import time

if __name__ == "__main__":
    # SPEC 2017 benchmarks
    spec_benchmarks = [
        500, 502, 505, 520, 523, 525, 531, 541, 548, 557,
        503, 507, 508, 510, 511, 519, 521, 527, 538, 544, 549, 554
    ]
    
    # DaCapo benchmarks
    dacapo_benchmarks = [
        'avrora', 'batik', 'biojava', 'cassandra', 'eclipse', 'fop', 'graphchi', 'h2', 'h2o', 'jme',
        'jython', 'kafka', 'luindex', 'lusearch', 'pmd', 'spring', 'sunflow', 'tomcat', 'tradebeans', 'tradesoap', 'xalan', 'zxing'
    ]
      
    # ARM DESKTOP EVENT GROUPS - 4 COUNTERS PER GROUP (TESTED FOR CONFLICTS)
    # Each group contains 4 events that work together without conflicts
    # Total: 13 groups covering 52 unique events (51 non-instruction + instructions in each group)
    
    # Group 0: Basic Performance Counters
    arm_group0 = "\"{instructions,cpu-cycles,branch-instructions,branch-misses}:S\""
    
    # Group 1: L1 Cache Performance
    arm_group1 = "\"{instructions,L1-dcache-loads,L1-dcache-load-misses,L1-icache-loads}:S\""
    
    # Group 2: Cache References and TLB Data
    arm_group2 = "\"{instructions,cache-misses,cache-references,dTLB-loads,dTLB-load-misses}:S\""
    
    # Group 3: TLB and System Events
    arm_group3 = "\"{instructions,iTLB-loads,iTLB-load-misses,context-switches}:S\""
    
    # Group 4: System Faults and Performance
    arm_group4 = "\"{instructions,page-faults,alignment-faults,emulation-faults}:S\""
    
    # Group 5: System Performance and Minor Faults
    arm_group5 = "\"{instructions,cpu-migrations,minor-faults,major-faults}:S\""
    
    # Group 6: ARM PMU TLB and Branch Events
    arm_group6 = "\"{instructions,dtlb_walk,itlb_walk,armv8_pmuv3_0/branch-loads/}:S\""
    
    # Group 7: ARM PMU Branch and L1 Data Cache
    arm_group7 = "\"{instructions,armv8_pmuv3_0/branch-load-misses/,armv8_pmuv3_0/L1-dcache-loads/,armv8_pmuv3_0/L1-dcache-load-misses/}:S\""
    
    # Group 8: ARM PMU L1 Instruction Cache and TLB
    arm_group8 = "\"{instructions,armv8_pmuv3_0/L1-icache-loads/,armv8_pmuv3_0/L1-icache-load-misses/,armv8_pmuv3_0/dTLB-loads/}:S\""
    
    # Group 9: ARM PMU TLB Performance
    arm_group9 = "\"{instructions,armv8_pmuv3_0/dTLB-load-misses/,armv8_pmuv3_0/iTLB-loads/,armv8_pmuv3_0/iTLB-load-misses/}:S\""
    
    # Group 10: System Time and Cache Events
    arm_group10 = "\"{instructions,system_time,l1d_cache,l1i_cache}:S\""
    
    # Group 11: Branch Performance
    arm_group11 = "\"{instructions,branches,branch-loads,branch-load-misses}:S\""
    
    # Group 12: Clock and System Events
    arm_group12 = "\"{instructions,task-clock,cpu-clock,cs}:S\""
    
    # Group 13: Pipeline Stall Events
    arm_group13 = "\"{instructions,bx_stall,fx_stall,ixa_stall}:S\""
    
    # Group 14: Additional Pipeline Stall Events
    arm_group14 = "\"{instructions,ixb_stall,lx_stall,decode_stall}:S\""
    
    # Group 15: Extended Pipeline Stall Events
    arm_group15 = "\"{instructions,dispatch_stall,sx_stall,cycles}:S\""
    
    # Group 16: Memory Operation Events
    arm_group16 = "\"{instructions,mem_access,mem_access_rd,mem_access_wr}:S\""
    
    # Group 17: Memory Error Events
    arm_group17 = "\"{instructions,memory_error,migrations,faults}:S\""

    # Combine all groups for easy iteration
    arm_groups = [
        arm_group0, arm_group1, arm_group2, arm_group3,
        arm_group4, arm_group5, arm_group6, arm_group7,
        arm_group8, arm_group9, arm_group10, arm_group11,
        arm_group12, arm_group13, arm_group14, arm_group15,
        arm_group16, arm_group17
    ]

    # ARM server configuration
    cpus = [0]  # Multiple CPU cores to test
    freq = '3.0GHz'
    
    # Granularity: 10M instructions
    granularities = [10000000]
    
    # DaCapo directory
    dacapo_dir = '../../../benchmarks/dacapo/'
    
    # SPEC directory  
    spec_dir = '/home/meb4744/PerfCount/benchmarks/spec'

    
    for cpu in cpus:
        for benchmark in spec_benchmarks:
            for granularity in granularities:
                count = 0
                for events in arm_groups:
                    command  = "perf record -a -C " + str(cpu)
                    command += " -e " + str(events)
                    command += " -c " + str(granularity)
                    command += " -o cpu_" + str(cpu) + "_" + str(freq) + "_spec_" + str(benchmark) + "_" + str(granularity) + "_" + str(count) + ".out"
                    command += " taskset --cpu-list " + str(cpu)
                    command += " bash -c 'cd " + spec_dir + " && source shrc && runcpu --base --config=matthew-1cpu " + str(benchmark) + "'"
                    print(command)
                    count += 1
    
    for cpu in cpus:
        for benchmark in dacapo_benchmarks:
            for granularity in granularities:
                count = 0
                for events in arm_groups:
                    command  = "perf record -a -C " + str(cpu)
                    command += " -e " + str(events)
                    command += " -c " + str(granularity)
                    command += " -o cpu_" + str(cpu) + "_" + str(freq) + "_dacapo_" + str(benchmark) + "_" + str(granularity) + "_" + str(count) + ".out"
                    command += " taskset --cpu-list " + str(cpu)
                    command += " java -jar " + dacapo_dir + "dacapo-23.11-MR2-chopin.jar -t 1 -n 10 " + str(benchmark)
                    print(command)
                    count += 1

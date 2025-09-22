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
      
    # Group 0: Basic Performance (Core execution)
    arm_group0 = "\"{instructions,cpu-cycles,branch-instructions,branch-misses}:S\""
    
    # Group 1: L1 Data Cache
    arm_group1 = "\"{instructions,L1-dcache-loads,L1-dcache-load-misses,cache-references}:S\""
    
    # Group 2: L1 Instruction Cache  
    arm_group2 = "\"{instructions,L1-icache-loads,L1-icache-load-misses,cache-misses}:S\""
    
    # Group 3: Data TLB
    arm_group3 = "\"{instructions,dTLB-loads,dTLB-load-misses,stalled-cycles-backend}:S\""
    
    # Group 4: Instruction TLB
    arm_group4 = "\"{instructions,iTLB-loads,iTLB-load-misses,stalled-cycles-frontend}:S\""
    
    # Group 5: Branch Prediction
    arm_group5 = "\"{instructions,branch-loads,branch-load-misses,bus-cycles}:S\""
    
    # Group 6: System Events
    arm_group6 = "\"{instructions,context-switches,cpu-migrations,page-faults}:S\""
    
    # Group 7: Memory and Fault Events
    arm_group7 = "\"{instructions,major-faults,minor-faults,alignment-faults}:S\""
    
    # Group 8: Software Clock Events
    arm_group8 = "\"{instructions,cpu-clock,task-clock,emulation-faults}:S\""
    
    # Group 9: Advanced System Events
    arm_group9 = "\"{instructions,cgroup-switches,bpf-output,dummy}:S\""
    
    # Group 10: ARMv8 PMU L1 Cache Events (verified working)
    arm_group10 = "\"{instructions,armv8_pmuv3_0/l1d_cache/,armv8_pmuv3_0/l1d_cache_refill/,armv8_pmuv3_0/l1d_cache_wb/}:S\""
    
    # Group 11: ARMv8 PMU L1 Instruction Cache Events
    arm_group11 = "\"{instructions,armv8_pmuv3_0/l1i_cache/,armv8_pmuv3_0/l1i_cache_refill/,armv8_pmuv3_0/bus_access/}:S\""
    
    # Group 12: ARMv8 PMU L2 Cache Events  
    arm_group12 = "\"{instructions,armv8_pmuv3_0/l2d_cache/,armv8_pmuv3_0/l2d_cache_refill/,armv8_pmuv3_0/l2d_cache_wb/}:S\""
    
    # Group 13: ARMv8 PMU L3 Cache Events (using available l3d_cache_wb)
    arm_group13 = "\"{instructions,armv8_pmuv3_0/l3d_cache_wb/,armv8_pmuv3_0/bus_cycles/,armv8_pmuv3_0/exc_taken/}:S\""
    
    # Group 14: ARMv8 PMU Memory Events
    arm_group14 = "\"{instructions,armv8_pmuv3_0/mem_access/,armv8_pmuv3_0/memory_error/,armv8_pmuv3_0/exc_return/}:S\""
    
    # Group 15: ARMv8 PMU Branch Events
    arm_group15 = "\"{instructions,armv8_pmuv3_0/br_pred/,armv8_pmuv3_0/br_mis_pred/,armv8_pmuv3_0/br_return_retired/}:S\""
    
    # Group 16: ARMv8 PMU Instruction Events
    arm_group16 = "\"{instructions,armv8_pmuv3_0/inst_retired/,armv8_pmuv3_0/op_retired/,armv8_pmuv3_0/op_spec/}:S\""
    
    # Group 17: ARMv8 PMU Load/Store Events
    arm_group17 = "\"{instructions,armv8_pmuv3_0/ld_retired/,armv8_pmuv3_0/st_retired/,armv8_pmuv3_0/ldst_spec/}:S\""
    
    # Group 18: ARMv8 PMU Advanced Events
    arm_group18 = "\"{instructions,armv8_pmuv3_0/dp_spec/,armv8_pmuv3_0/ase_spec/,armv8_pmuv3_0/vfp_spec/}:S\""

    # Combine all groups for easy iteration
    arm_groups = [
        arm_group0, arm_group1, arm_group2, arm_group3,
        arm_group4, arm_group5, arm_group6, arm_group7,
        arm_group8, arm_group9, arm_group10, arm_group11,
        arm_group12, arm_group13, arm_group14, arm_group15,
        arm_group16, arm_group17, arm_group18
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
                    command += " bash -c 'cd " + spec_dir + " && source shrc && runcpu -c base --config=matthew-1cpu " + str(benchmark) + "'"
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

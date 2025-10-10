#!/usr/bin/env python3

import os
import stat
import time

if __name__ == "__main__":
    # Output file for shell script
    output_file = "collection.sh"
    
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
      
    # ARM Edge Heterogeneous Platform - 2-counter groups (instructions + unique counter)
    # Based on available ARMv8 PMU events, selecting diverse counters that show unique behavior
    
    # Group 0: Basic Performance - CPU Cycles
    perf_events_group0 = "instructions,cpu-cycles"
    
    # Group 1: Branch Prediction Performance
    perf_events_group1 = "instructions,armv8_pmuv3/br_retired/"
    
    # Group 2: Branch Misprediction
    perf_events_group2 = "instructions,armv8_pmuv3/br_mis_pred/"
    
    # Group 3: Instruction Retirement
    perf_events_group3 = "instructions,armv8_pmuv3/inst_retired/"
    
    # Group 4: Instruction Speculation
    perf_events_group4 = "instructions,armv8_pmuv3/inst_spec/"
    
    # Group 5: L1 Data Cache Access
    perf_events_group5 = "instructions,armv8_pmuv3/l1d_cache/"
    
    # Group 6: L1 Data Cache Refill (Misses)
    perf_events_group6 = "instructions,armv8_pmuv3/l1d_cache_refill/"
    
    # Group 7: L1 Data Cache Writeback
    perf_events_group7 = "instructions,armv8_pmuv3/l1d_cache_wb/"
    
    # Group 8: L1 Data TLB Access
    perf_events_group8 = "instructions,armv8_pmuv3/l1d_tlb/"
    
    # Group 9: L1 Data TLB Refill (Misses)
    perf_events_group9 = "instructions,armv8_pmuv3/l1d_tlb_refill/"
    
    # Group 10: L1 Instruction Cache Access
    perf_events_group10 = "instructions,armv8_pmuv3/l1i_cache/"
    
    # Group 11: L1 Instruction Cache Refill (Misses)
    perf_events_group11 = "instructions,armv8_pmuv3/l1i_cache_refill/"
    
    # Group 12: L1 Instruction TLB Access
    perf_events_group12 = "instructions,armv8_pmuv3/l1i_tlb/"
    
    # Group 13: L1 Instruction TLB Refill (Misses)
    perf_events_group13 = "instructions,armv8_pmuv3/l1i_tlb_refill/"
    
    # Group 14: L2 Data Cache Access
    perf_events_group14 = "instructions,armv8_pmuv3/l2d_cache/"
    
    # Group 15: L2 Data Cache Refill (Misses)
    perf_events_group15 = "instructions,armv8_pmuv3/l2d_cache_refill/"
    
    # Group 16: L2 Data Cache Writeback
    perf_events_group16 = "instructions,armv8_pmuv3/l2d_cache_wb/"
    
    # Group 17: L2 Data TLB Access
    perf_events_group17 = "instructions,armv8_pmuv3/l2d_tlb/"
    
    # Group 18: L2 Data TLB Refill (Misses)
    perf_events_group18 = "instructions,armv8_pmuv3/l2d_tlb_refill/"
    
    # Group 19: L3 Data Cache Access
    perf_events_group19 = "instructions,armv8_pmuv3/l3d_cache/"
    
    # Group 20: L3 Data Cache Refill (Misses)
    perf_events_group20 = "instructions,armv8_pmuv3/l3d_cache_refill/"
    
    # Group 21: L3 Data Cache Writeback
    perf_events_group21 = "instructions,armv8_pmuv3/l3d_cache_wb/"
    
    # Group 22: Memory Access
    perf_events_group22 = "instructions,armv8_pmuv3/mem_access/"
    
    # Group 23: Backend Stall
    perf_events_group23 = "instructions,armv8_pmuv3/stall_backend/"
    
    # Group 24: Frontend Stall
    perf_events_group24 = "instructions,armv8_pmuv3/stall_frontend/"
    
    # Group 25: Bus Access
    perf_events_group25 = "instructions,armv8_pmuv3/bus_access/"
    
    # Group 26: Bus Cycles
    perf_events_group26 = "instructions,armv8_pmuv3/bus_cycles/"
    
    # Group 27: Exception Taken
    perf_events_group27 = "instructions,armv8_pmuv3/exc_taken/"
    
    # Group 28: Exception Return
    perf_events_group28 = "instructions,armv8_pmuv3/exc_return/"
    
    # Group 29: Memory Error
    perf_events_group29 = "instructions,armv8_pmuv3/memory_error/"
    
    # Group 30: Software Increment
    perf_events_group30 = "instructions,armv8_pmuv3/sw_incr/"
    
    # Group 31: Context ID Write Retired
    perf_events_group31 = "instructions,armv8_pmuv3/cid_write_retired/"
    
    # Group 32: TTBR Write Retired
    perf_events_group32 = "instructions,armv8_pmuv3/ttbr_write_retired/"
    
    # Group 33: Branch Prediction
    perf_events_group33 = "instructions,armv8_pmuv3/br_pred/"
    
    # Group 34: Branch Misprediction Retired
    perf_events_group34 = "instructions,armv8_pmuv3/br_mis_pred_retired/"
    
    # Group 35: L2 Data Cache Allocate
    perf_events_group35 = "instructions,armv8_pmuv3/l2d_cache_allocate/"
    
    # Group 36: L3 Data Cache Allocate
    perf_events_group36 = "instructions,armv8_pmuv3/l3d_cache_allocate/"
    
    # Additional Hardware Events (Generic ARM counters)
    # Group 37: Generic Branch Misses
    perf_events_group37 = "instructions,branch-misses"
    
    # Group 38: Generic Bus Cycles
    perf_events_group38 = "instructions,bus-cycles"
    
    # Group 39: Generic Cache Misses
    perf_events_group39 = "instructions,cache-misses"
    
    # Group 40: Generic Cache References
    perf_events_group40 = "instructions,cache-references"
    
    # Group 41: Generic Backend Stall Cycles
    perf_events_group41 = "instructions,stalled-cycles-backend"
    
    # Group 42: Generic Frontend Stall Cycles
    perf_events_group42 = "instructions,stalled-cycles-frontend"
    
    # Software Events (System-level performance)
    # Group 43: Context Switches
    perf_events_group43 = "instructions,context-switches"
    
    # Group 44: Page Faults
    perf_events_group44 = "instructions,page-faults"
    
    # Group 45: Major Page Faults
    perf_events_group45 = "instructions,major-faults"
    
    # Group 46: Minor Page Faults
    perf_events_group46 = "instructions,minor-faults"
    
    # Combine all groups for easy iteration
    perf_events_groups = [
        perf_events_group0, perf_events_group1, perf_events_group2, perf_events_group3,
        perf_events_group4, perf_events_group5, perf_events_group6, perf_events_group7,
        perf_events_group8, perf_events_group9, perf_events_group10, perf_events_group11,
        perf_events_group12, perf_events_group13, perf_events_group14, perf_events_group15,
        perf_events_group16, perf_events_group17, perf_events_group18, perf_events_group19,
        perf_events_group20, perf_events_group21, perf_events_group22, perf_events_group23,
        perf_events_group24, perf_events_group25, perf_events_group26, perf_events_group27,
        perf_events_group28, perf_events_group29, perf_events_group30, perf_events_group31,
        perf_events_group32, perf_events_group33, perf_events_group34, perf_events_group35,
        perf_events_group36, perf_events_group37, perf_events_group38, perf_events_group39,
        perf_events_group40, perf_events_group41, perf_events_group42, perf_events_group43,
        perf_events_group44, perf_events_group45, perf_events_group46
    ]

    # CPU mapping for ARM Edge Heterogeneous Platform
    # taskset 0: InO core
    # taskset 4: OOO core  
    # taskset 7: OOO_cache core
    # cpus = [0, 4, 7]
    cpus = [0, 4] 
    freq = '1.5GHz'
    
    # Granularity: 10M instructions
    granularities = [10000000]
    
    # DaCapo directory
    dacapo_dir = '../../../benchmarks/dacapo/'
    
    # SPEC directory  
    spec_dir = '/home/meb4744/PerfCount/benchmarks/spec'

    # Open output file and write shebang
    with open(output_file, 'w') as f:
        f.write("#!/bin/bash\n\n")
        
        # Generate SPEC benchmark commands
        for cpu in cpus:
            for benchmark in spec_benchmarks:
                for granularity in granularities:
                    count = 0
                    for events in perf_events_groups:
                        command  = "perf record -a -C " + str(cpu)
                        command += " -e " + str(events)
                        command += " -c " + str(granularity)
                        command += " --no-buffering"
                        command += " -o cpu_" + str(cpu) + "_" + str(freq) + "_spec_" + str(benchmark) + "_" + str(granularity) + "_" + str(count) + ".out"
                        command += " taskset --cpu-list " + str(cpu)
                        command += " bash -c 'cd " + spec_dir + " && source shrc && runcpu -c base --config=matthew-1cpu " + str(benchmark) + "'"
                        f.write(command + "\n")
                        count += 1
        
        # Generate DaCapo benchmark commands
        for cpu in cpus:
            for benchmark in dacapo_benchmarks:
                for granularity in granularities:
                    count = 0
                    for events in perf_events_groups:
                        command  = "perf record -a -C " + str(cpu)
                        command += " -e " + str(events)
                        command += " -c " + str(granularity)
                        command += " --no-buffering"
                        command += " -o cpu_" + str(cpu) + "_" + str(freq) + "_dacapo_" + str(benchmark) + "_" + str(granularity) + "_" + str(count) + ".out"
                        command += " taskset --cpu-list " + str(cpu)
                        command += " java -XX:+UseSerialGC -Xint -XX:ParallelGCThreads=1 -XX:CICompilerCount=1 -XX:-BackgroundCompilation -XX:-TieredCompilation -Xms2g -Xmx2g -XX:-UseAdaptiveSizePolicy -jar " + dacapo_dir + "dacapo-23.11-MR2-chopin.jar -t 1 -n 1 " + str(benchmark)
                        f.write(command + "\n")
                        count += 1
        
        # Add email notification at the end of the script
        f.write("\n")
        f.write("# Send email notification upon completion\n")
        f.write("END_TIME=$(date)\n")
        f.write("HOSTNAME=$(hostname)\n")
        f.write("EMAIL_SENDER=\"mebarondeau@utexas.edu\"\n")
        f.write("EMAIL_RECIPIENT=\"mebarondeau@utexas.edu\"\n")
        f.write("EMAIL_SUBJECT=\"PerfCount Data Collection Completed on $HOSTNAME\"\n")
        f.write("EMAIL_BODY=\"The data collection workloads have completed successfully.\n")
        f.write("\n")
        f.write("Hostname: $HOSTNAME\n")
        f.write("End Time: $END_TIME\n")
        f.write("Script: arm_edge_heterogeneous/collection.sh\n")
        f.write("Working Directory: $(pwd)\n")
        f.write("\n")
        f.write("All DaCapo and SPEC benchmarks have been processed.\n")
        f.write("\"\n")
        f.write("\n")
        f.write("echo \"$EMAIL_BODY\" | mail -s \"$EMAIL_SUBJECT\" -r \"$EMAIL_SENDER\" \"$EMAIL_RECIPIENT\"\n")
        f.write("\n")
        f.write("echo \"Data collection completed at $END_TIME\"\n")
        f.write("echo \"Email notification sent to $EMAIL_RECIPIENT\"\n")
    
    # Make the script executable
    os.chmod(output_file, os.stat(output_file).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    
    print(f"\nGenerated {output_file} successfully!")
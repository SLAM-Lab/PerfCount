#!/usr/bin/env python3

import os
import stat

if __name__ == "__main__":
    output_file = "dacapo_collection.sh"
   
    dacapo_benchmarks = [
        'avrora', 'batik', 'biojava', 'cassandra', 'eclipse', 'fop', 'graphchi', 'h2', 'h2o', 'jme',
        'jython', 'kafka', 'luindex', 'lusearch', 'pmd', 'spring', 'sunflow', 'tomcat', 'tradebeans', 'tradesoap', 'xalan', 'zxing'
    ]

    frequencies = ["4.0"] 
    p_cpus = [0]
    e_cpus = [16]
    dacapo_dir = "../../../benchmarks/dacapo"

    # 8 P-Core Groups
    p_groups = [
        # 0. Baseline & Compute
        "{cpu_core/instructions/,cpu_core/cpu-cycles/,cpu_core/bus-cycles/,cpu_core/fp_arith_inst_retired.scalar_single/}:uS",
        
        # 1. General Branches & Page Faults
        "{cpu_core/instructions/,cpu_core/branch-loads/,cpu_core/branch-load-misses/}:uS",
        
        # 2. Retired Branches & Frequency Scaling
        "{cpu_core/instructions/,cpu_core/br_inst_retired.all_branches/,cpu_core/br_misp_retired.all_branches/,cpu_core/ref-cycles/}:uS",
        
        # 3. L1 Data Cache
        "{cpu_core/instructions/,cpu_core/L1-dcache-loads/,cpu_core/L1-dcache-load-misses/,cpu_core/L1-dcache-stores/}:uS",
        
        # 4. L1 Instruction Cache & Last Level Cache (LLC)
        "{cpu_core/instructions/,cpu_core/L1-icache-load-misses/,cpu_core/LLC-loads/,cpu_core/LLC-load-misses/}:uS",
        
        # 5. Generic Cache & Memory Loads
        "{cpu_core/instructions/,cpu_core/cache-references/,cpu_core/cache-misses/,cpu_core/mem-loads/}:uS",
        
        # 6. TLB Reads
        "{cpu_core/instructions/,cpu_core/dTLB-loads/,cpu_core/dTLB-load-misses/,cpu_core/iTLB-load-misses/}:uS",
        
        # 7. TLB Writes & Memory Stores
        "{cpu_core/instructions/,cpu_core/dTLB-stores/,cpu_core/dTLB-store-misses/,cpu_core/mem-stores/}:uS"
    ]

    # 8 E-Core Groups
    e_groups = [
        # 0. Baseline & Compute
        "{cpu_atom/instructions/,cpu_atom/cpu-cycles/,cpu_atom/bus-cycles/,cpu_atom/fp_arith_inst_retired.scalar_single/}:uS",
        
        # 1. General Branches & Page Faults
        "{cpu_atom/instructions/,cpu_atom/branch-loads/,cpu_atom/branch-load-misses/}:uS",
        
        # 2. Retired Branches & Frequency Scaling
        "{cpu_atom/instructions/,cpu_atom/br_inst_retired.all_branches/,cpu_atom/br_misp_retired.all_branches/,cpu_atom/ref-cycles/}:uS",
        
        # 3. L1 Data Cache
        "{cpu_atom/instructions/,cpu_atom/L1-dcache-loads/,cpu_atom/L1-dcache-load-misses/,cpu_atom/L1-dcache-stores/}:uS",
        
        # 4. L1 Instruction Cache & Last Level Cache (LLC)
        "{cpu_atom/instructions/,cpu_atom/L1-icache-load-misses/,cpu_atom/LLC-loads/,cpu_atom/LLC-load-misses/}:uS",
        
        # 5. Generic Cache & Memory Loads
        "{cpu_atom/instructions/,cpu_atom/cache-references/,cpu_atom/cache-misses/,cpu_atom/mem-loads/}:uS",
        
        # 6. TLB Reads
        "{cpu_atom/instructions/,cpu_atom/dTLB-loads/,cpu_atom/dTLB-load-misses/,cpu_atom/iTLB-load-misses/}:uS",
        
        # 7. TLB Writes & Memory Stores
        "{cpu_atom/instructions/,cpu_atom/dTLB-stores/,cpu_atom/dTLB-store-misses/,cpu_atom/mem-stores/}:uS"
    ]

    

    with open(output_file, "w") as f:
        f.write("#!/bin/bash\n\n")
        
        # --- DaCapo P-Core ---
        for i, group in enumerate(p_groups):
            for freq in frequencies:
                for cpu in p_cpus:
                    for benchmark in dacapo_benchmarks:
                        command = f"perf record -a -C {cpu} -e \"{group}\" -c 10000000 --no-buffering -o cpu_{cpu}_{freq}GHz_dacapo_{benchmark}_10000000_{i}_0.out -- "
                        command += f"taskset --cpu-list {cpu} java -Xcomp -XX:+UseSerialGC -Xms2g -Xmx2g -XX:-UseAdaptiveSizePolicy -jar {dacapo_dir}/dacapo-23.11-MR2-chopin.jar -s large -t 1 -n 1 {benchmark}"
                        f.write(command + "\n") 

        # --- DaCapo E-Core ---
        for i, group in enumerate(e_groups):
            for freq in frequencies:
                for cpu in e_cpus:
                    for benchmark in dacapo_benchmarks:
                        command = f"perf record -a -C {cpu} -e \"{group}\" -c 10000000 --no-buffering -o cpu_{cpu}_{freq}GHz_dacapo_{benchmark}_10000000_{i}_0.out -- "
                        command += f"taskset --cpu-list {cpu} java -Xcomp -XX:+UseSerialGC -Xms2g -Xmx2g -XX:-UseAdaptiveSizePolicy -jar {dacapo_dir}/dacapo-23.11-MR2-chopin.jar -s large -t 1 -n 1 {benchmark}"
                        f.write(command + "\n") 

        f.write("\necho \"DaCapo x86 Data collection completed\" | mail -s \"DaCapo Complete\" -r \"mebarondeau@utexas.edu\" \"mebarondeau@utexas.edu\"\n")

    st = os.stat(output_file)
    os.chmod(output_file, st.st_mode | stat.S_IEXEC)
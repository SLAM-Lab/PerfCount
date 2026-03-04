#!/usr/bin/env python3

import os
import stat

if __name__ == "__main__":
    output_file = "dacapo_collection.sh"
   
    # DaCapo benchmarks
    dacapo_benchmarks = [
        'avrora', 'batik', 'biojava', 'cassandra', 'eclipse', 'fop', 'graphchi', 'h2', 'h2o', 'jme',
        'jython', 'kafka', 'luindex', 'lusearch', 'pmd', 'spring', 'sunflow', 'tomcat', 'tradebeans', 'tradesoap', 'xalan', 'zxing'
    ]

    frequencies = ["3.0"] 
    cpus = [0]
    dacapo_dir = "../../../benchmarks/dacapo"

    # 18 Hardware Counter Groups for ARM Desktop (User-Space Only)
    arm_groups = [
        "{instructions,cpu-cycles,branch-instructions,branch-misses}:Su",
        "{instructions,L1-dcache-loads,L1-dcache-load-misses,L1-icache-loads}:Su",
        "{instructions,cache-misses,cache-references,dTLB-loads,dTLB-load-misses}:Su",
        "{instructions,iTLB-loads,iTLB-load-misses,context-switches}:Su",
        "{instructions,page-faults,alignment-faults,emulation-faults}:Su",
        "{instructions,cpu-migrations,minor-faults,major-faults}:Su",
        "{instructions,dtlb_walk,itlb_walk,armv8_pmuv3_0/branch-loads/}:Su",
        "{instructions,armv8_pmuv3_0/branch-load-misses/,armv8_pmuv3_0/L1-dcache-loads/,armv8_pmuv3_0/L1-dcache-load-misses/}:Su",
        "{instructions,armv8_pmuv3_0/L1-icache-loads/,armv8_pmuv3_0/L1-icache-load-misses/,armv8_pmuv3_0/dTLB-loads/}:Su",
        "{instructions,armv8_pmuv3_0/dTLB-load-misses/,armv8_pmuv3_0/iTLB-loads/,armv8_pmuv3_0/iTLB-load-misses/}:Su",
        "{instructions,system_time,l1d_cache,l1i_cache}:Su",
        "{instructions,branches,branch-loads,branch-load-misses}:Su",
        "{instructions,task-clock,cpu-clock,cs}:Su",
        "{instructions,bx_stall,fx_stall,ixa_stall}:Su",
        "{instructions,ixb_stall,lx_stall,decode_stall}:Su",
        "{instructions,dispatch_stall,sx_stall,cycles}:Su",
        "{instructions,mem_access,mem_access_rd,mem_access_wr}:Su",
        "{instructions,memory_error,migrations,faults}:Su"
    ]

    with open(output_file, "w") as f:
        f.write("#!/bin/bash\n\n")
        count = 0
        
        for counter_group in range(0, len(arm_groups)):
            for freq in frequencies:
                for cpu in cpus:
                    for benchmark in dacapo_benchmarks:
                        # 1. Start perf system-wide (-a) on the target CPU (-C cpu)
                        command = f"perf record -a -C {cpu} "
                        command += f"-e \"{arm_groups[counter_group]}\" "
                        command += "-c 10000000 --no-buffering -o "
                        command += f"cpu_{cpu}_{freq}GHz_dacapo_{benchmark}_10000000_{counter_group}_0.out -- "
                        
                        # 2. Run the Java command pinned to the same CPU with -Xcomp.
                        command += f"taskset --cpu-list {cpu} "
                        command += f"java -Xcomp -XX:+UseSerialGC -Xms2g -Xmx2g -XX:-UseAdaptiveSizePolicy -jar {dacapo_dir}/dacapo-23.11-MR2-chopin.jar -t 1 -n 1 -s large {benchmark}"
                        
                        f.write(command + "\n") 
                        count += 1

        # Email notification block
        f.write("\n")
        f.write("END_TIME=$(date)\n")
        f.write("HOSTNAME=$(hostname)\n")
        f.write("echo \"DaCapo Data collection completed at $END_TIME on $HOSTNAME\" | mail -s \"DaCapo Collection Complete\" -r \"mebarondeau@utexas.edu\" \"mebarondeau@utexas.edu\"\n")

    st = os.stat(output_file)
    os.chmod(output_file, st.st_mode | stat.S_IEXEC)
    
    print(f"Generated {output_file} with {count} DaCapo commands using Per-CPU monitoring and -Xcomp.")

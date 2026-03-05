#!/usr/bin/env python3

import os
import stat

if __name__ == "__main__":
    output_file = "dacapo_collection.sh"
   
    dacapo_benchmarks = [
        'avrora', 'batik', 'biojava', 'cassandra', 'eclipse', 'fop', 'graphchi', 'h2', 'h2o', 'jme',
        'jython', 'kafka', 'luindex', 'lusearch', 'pmd', 'spring', 'sunflow', 'tomcat', 'tradebeans', 'tradesoap', 'xalan', 'zxing'
    ]

    # Heterogeneous Edge Setup
    frequencies = ["1.0"]
    cpus = [0, 4, 7]
    dacapo_dir = "../../../benchmarks/dacapo"

    # 20 Hardware Counter Groups for ARM Edge (User-Space Only)
    arm_groups = [
        "{instructions,cpu-cycles}:Su",
        "{instructions,branch-instructions}:Su",
        "{instructions,branch-misses}:Su",
        "{instructions,L1-dcache-loads}:Su",
        "{instructions,L1-dcache-load-misses}:Su",
        "{instructions,L1-icache-loads}:Su",
        "{instructions,L1-icache-load-misses}:Su",
        "{instructions,cache-references}:Su",
        "{instructions,cache-misses}:Su",
        "{instructions,dTLB-loads}:Su",
        "{instructions,dTLB-load-misses}:Su",
        "{instructions,iTLB-loads}:Su",
        "{instructions,iTLB-load-misses}:Su",
        "{instructions,context-switches}:Su",
        "{instructions,page-faults}:Su",
        "{instructions,alignment-faults}:Su",
        "{instructions,emulation-faults}:Su",
        "{instructions,cpu-migrations}:Su",
        "{instructions,minor-faults}:Su",
        "{instructions,major-faults}:Su"
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
                        
                        # 2. Run the Java command pinned to the same CPU with -Xcomp and -s large
                        command += f"taskset --cpu-list {cpu} "
                        command += f"java -Xcomp -XX:+UseSerialGC -Xms2g -Xmx2g -XX:-UseAdaptiveSizePolicy -jar {dacapo_dir}/dacapo-23.11-MR2-chopin.jar -s large -t 1 -n 1 {benchmark}"
                        
                        f.write(command + "\n") 
                        count += 1

        # Email notification block
        f.write("\n")
        f.write("END_TIME=$(date)\n")
        f.write("HOSTNAME=$(hostname)\n")
        f.write("echo \"DaCapo Edge Data collection completed at $END_TIME on $HOSTNAME\" | mail -s \"DaCapo Collection Complete\" -r \"mebarondeau@utexas.edu\" \"mebarondeau@utexas.edu\"\n")

    st = os.stat(output_file)
    os.chmod(output_file, st.st_mode | stat.S_IEXEC)
    
    print(f"Generated {output_file} with {count} DaCapo commands using Per-CPU monitoring, -Xcomp, and -s large.")
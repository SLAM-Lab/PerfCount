#!/usr/bin/env python3

import os
import stat
import time

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

    # Counter groups with :Su to ensure we only capture user-space and avoid OS/Kernel noise
    counter_group_0 = "{instructions,cpu-cycles,stalled-cycles-backend,stalled-cycles-frontend}:Su"    
    counter_group_1 = "{instructions,armv8_pmuv3_0/br_pred/,armv8_pmuv3_0/br_mis_pred/,L1-dcache-loads}:Su"
    counter_group_2 = "{instructions,L1-dcache-load-misses,armv8_pmuv3_0/l1d_cache/,armv8_pmuv3_0/l1d_cache_refill/}:Su"
    counter_group_3 = "{instructions,armv8_pmuv3_0/l1d_cache_wb/,L1-icache-loads,L1-icache-load-misses}:Su"
    counter_group_4 = "{instructions,armv8_pmuv3_0/l1i_cache/,armv8_pmuv3_0/l1i_cache_refill/,armv8_pmuv3_0/l2d_cache/}:Su"
    counter_group_5 = "{instructions,armv8_pmuv3_0/l2d_cache_refill/,armv8_pmuv3_0/l2d_cache_wb/,cache-references}:Su"
    counter_group_6 = "{instructions,cache-misses,dTLB-loads,dTLB-load-misses}:Su"
    counter_group_7 = "{instructions,iTLB-loads,iTLB-load-misses,armv8_pmuv3_0/bus_access/}:Su"
    counter_group_8 = "{instructions,armv8_pmuv3_0/mem_access/,armv8_pmuv3_0/memory_error/,armv8_pmuv3_0/exc_return/}:Su"

    counter_groups = [
        counter_group_0, counter_group_1, counter_group_2, 
        counter_group_3, counter_group_4, counter_group_5,
        counter_group_6, counter_group_7, counter_group_8
    ]    

    with open(output_file, "w") as f:
        count = 0
        
        for counter_group in range(0,len(counter_groups)):
            for freq in frequencies:
                for cpu in cpus:
                    for benchmark in dacapo_benchmarks:
                        # 1. Start perf system-wide (-a) on the target CPU (-C cpu)
                        command = "perf record -a -C " + str(cpu)
                        command += " -e \"" + str(counter_groups[counter_group]) + "\""
                        command += " -c 10000000 --no-buffering -o "
                        command += "cpu_" +  str(cpu) + "_" + str(freq) + "GHz_dacapo_" 
                        command += str(benchmark) + "_10000000_" + str(counter_group) + "_0.out -- "
                        
                        # 2. Run the Java command pinned to the same CPU.
                        command += "taskset --cpu-list " + str(cpu) + " "
                        command += "java -Xcomp -XX:+UseSerialGC -Xms2g -Xmx2g -XX:-UseAdaptiveSizePolicy -jar " + dacapo_dir + "/dacapo-23.11-MR2-chopin.jar -t 1 -n 1 " + str(benchmark)
                        
                        f.write(command + "\n") 
                        count += 1

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
        f.write("Script: arm_server/dacapo_collection.sh\n")
        f.write("Working Directory: $(pwd)\n")
        f.write("\n")
        f.write("All DaCapo benchmarks have been processed.\n")
        f.write("\"\n")
        f.write("\n")
        f.write("echo \"$EMAIL_BODY\" | mail -s \"$EMAIL_SUBJECT\" -r \"$EMAIL_SENDER\" \"$EMAIL_RECIPIENT\"\n")
        f.write("\n")
        f.write("echo \"Data collection completed at $END_TIME\"\n")

    st = os.stat(output_file)
    os.chmod(output_file, st.st_mode | stat.S_IEXEC)
    
    print(f"Generated {output_file} with {count} DaCapo commands using Per-CPU monitoring.")
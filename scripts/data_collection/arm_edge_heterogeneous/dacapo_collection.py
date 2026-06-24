#!/usr/bin/env python3

import os
import stat

if __name__ == "__main__":
    output_file = "dacapo_collection.sh"
   
    # Benchmarks that can survive the absolute strictest flags
    strict_benchmarks = [
        'avrora', 'batik', 'biojava', 'eclipse', 'fop', 'graphchi', 'h2', 'jme', 
        'luindex', 'lusearch', 'pmd', 'sunflow', 'xalan', 'zxing'
    ]
    
    # Massive enterprise workloads that crash under extreme constraints (e.g., spring)
    relaxed_benchmarks = [
        'cassandra', 'h2o', 'jython', 'kafka', 'spring', 'tomcat', 'tradebeans', 'tradesoap'
    ]

    dacapo_benchmarks = strict_benchmarks + relaxed_benchmarks

    frequencies = ["2.0"] 
    cpus = [4]
    dacapo_dir = "../../../benchmarks/dacapo"

    # Strict Flags (Maximum Determinism ported from x86)
    strict_flags = (
        "-Xcomp -Xbatch -XX:-TieredCompilation -XX:CICompilerCount=1 "
        "-XX:+UnlockExperimentalVMOptions -XX:+UseSerialGC -XX:+AlwaysPreTouch "
        "-Xms4g -Xmx4g -XX:InitialCodeCacheSize=256m -XX:ReservedCodeCacheSize=256m "
        "-XX:+DisableExplicitGC -XX:-UseBiasedLocking -XX:-UsePerfData -XX:-UseTLAB "
        "-XX:+UnlockDiagnosticVMOptions -XX:GuaranteedSafepointInterval=0"
    )

    # Relaxed Flags (Removed UseTLAB to prevent framework crashes)
    relaxed_flags = (
        "-Xcomp -Xbatch -XX:-TieredCompilation -XX:CICompilerCount=1 "
        "-XX:+UnlockExperimentalVMOptions -XX:+UseSerialGC -XX:+AlwaysPreTouch "
        "-Xms4g -Xmx4g -XX:InitialCodeCacheSize=256m -XX:ReservedCodeCacheSize=256m "
        "-XX:+DisableExplicitGC -XX:-UseBiasedLocking -XX:-UsePerfData "
        "-XX:+UnlockDiagnosticVMOptions -XX:GuaranteedSafepointInterval=0"
    )

    arm_groups = [
        "{instructions,cpu-cycles}:Su",
#        "{instructions,branch-misses}:Su",
#        "{instructions,cache-misses}:Su",
#        "{instructions,cache-references}:Su",
#        "{instructions,stalled-cycles-backend}:Su",
#        "{instructions,stalled-cycles-frontend}:Su",
#        "{instructions,bus-cycles}:Su",
#        "{instructions,L1-dcache-loads}:Su",
#        "{instructions,L1-dcache-load-misses}:Su",
#        "{instructions,L1-icache-loads}:Su",
#        "{instructions,L1-icache-load-misses}:Su",
#        "{instructions,branch-loads}:Su",
#        "{instructions,branch-load-misses}:Su",
#        "{instructions,dTLB-loads}:Su",
#        "{instructions,dTLB-load-misses}:Su",
#        "{instructions,iTLB-loads}:Su",
#        "{instructions,iTLB-load-misses}:Su",
#        "{instructions,armv8_pmuv3/l2d_cache/}:Su",
#        "{instructions,armv8_pmuv3/l2d_cache_refill/}:Su",
#        "{instructions,armv8_pmuv3/l3d_cache/}:Su",
#        "{instructions,armv8_pmuv3/l3d_cache_refill/}:Su",
#        "{instructions,armv8_pmuv3/mem_access/}:Su",
#        "{instructions,page-faults}:Su",
#        "{instructions,armv8_pmuv3/br_retired/}:Su",
#        "{instructions,armv8_pmuv3/l1d_cache_wb/}:Su"
    ]

    script_dir = os.path.dirname(os.path.abspath(__file__))
    wrapper = os.path.join(script_dir, "arm_power_wrapper.sh")

    with open(output_file, "w") as f:
        f.write("#!/bin/bash\n\n")
        count = 0

        for counter_group in range(0, len(arm_groups)):
            for freq in frequencies:
                for cpu in cpus:
                    for benchmark in dacapo_benchmarks:

                        # Dynamically assign strict vs relaxed flags based on the benchmark
                        flags_to_use = strict_flags if benchmark in strict_benchmarks else relaxed_flags

                        base = f"cpu_{cpu}_{freq}GHz_dacapo_{benchmark}_10000000_{counter_group}_0"
                        command = f"{wrapper} {base} {cpu} \"{arm_groups[counter_group]}\" -- "
                        command += f"taskset --cpu-list {cpu} java {flags_to_use} -jar {dacapo_dir}/dacapo-23.11-MR2-chopin.jar -s default -t 1 -n 1 -f 5 {benchmark}"

                        f.write(command + "\n")
                        count += 1

        # Email notification block
        f.write("\n")
        f.write("END_TIME=$(date)\n")
        f.write("HOSTNAME=$(hostname)\n")
        f.write("echo \"DaCapo Data collection completed at $END_TIME on $HOSTNAME\" | mail -s \"DaCapo Collection Complete\" -r \"meb4744@cs.utexas.edu\" \"mebarondeau@utexas.edu\"\n")

    st = os.stat(output_file)
    os.chmod(output_file, st.st_mode | stat.S_IEXEC)
    
    print(f"Generated {output_file} with {count} DaCapo commands using highly deterministic JVM flags.")

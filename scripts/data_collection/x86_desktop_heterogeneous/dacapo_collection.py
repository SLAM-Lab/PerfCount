#!/usr/bin/env python3

import os
import stat

if __name__ == "__main__":
    output_file = "dacapo_collection.sh"
    
    email_address = "mebarondeau@utexas.edu"
   
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

    frequencies = ["4.0"] 
    p_cpus = [0]
    e_cpus = [16]
    dacapo_dir = "../../../benchmarks/dacapo"

    # Strict Flags (Maximum Determinism)
    strict_flags = (
        "-Xcomp -Xbatch -XX:-TieredCompilation -XX:CICompilerCount=1 "
        "-XX:+UnlockExperimentalVMOptions -XX:+UseEpsilonGC -XX:+AlwaysPreTouch "
        "-Xms16g -Xmx16g -XX:InitialCodeCacheSize=256m -XX:ReservedCodeCacheSize=256m "
        "-XX:+DisableExplicitGC -XX:-UseBiasedLocking -XX:-UsePerfData -XX:-UseTLAB "
        "-XX:+UnlockDiagnosticVMOptions -XX:GuaranteedSafepointInterval=0"
    )

    # Relaxed Flags (Removed hashCode=2 and UseTLAB to prevent framework crashes)
    relaxed_flags = (
        "-Xcomp -Xbatch -XX:-TieredCompilation -XX:CICompilerCount=1 "
        "-XX:+UnlockExperimentalVMOptions -XX:+UseEpsilonGC -XX:+AlwaysPreTouch "
        "-Xms16g -Xmx16g -XX:InitialCodeCacheSize=256m -XX:ReservedCodeCacheSize=256m "
        "-XX:+DisableExplicitGC -XX:-UseBiasedLocking -XX:-UsePerfData "
        "-XX:+UnlockDiagnosticVMOptions -XX:GuaranteedSafepointInterval=0"
    )

    # 8 P-Core Groups (Updated with exact casing)
    p_groups = [
        "{cpu_core/instructions/,cpu_core/cpu-cycles/,cpu_core/bus-cycles/,cpu_core/fp_arith_inst_retired.scalar_single/}:uS",
        "{cpu_core/instructions/,cpu_core/branch-loads/,cpu_core/branch-load-misses/}:uS",
        "{cpu_core/instructions/,cpu_core/br_inst_retired.all_branches/,cpu_core/br_misp_retired.all_branches/,cpu_core/ref-cycles/}:uS",
        "{cpu_core/instructions/,cpu_core/L1-dcache-loads/,cpu_core/L1-dcache-load-misses/,cpu_core/L1-dcache-stores/}:uS",
        "{cpu_core/instructions/,cpu_core/L1-icache-load-misses/,cpu_core/LLC-loads/,cpu_core/LLC-load-misses/}:uS",
        "{cpu_core/instructions/,cpu_core/cache-references/,cpu_core/cache-misses/,cpu_core/mem-loads/}:uS",
        "{cpu_core/instructions/,cpu_core/dTLB-loads/,cpu_core/dTLB-load-misses/,cpu_core/iTLB-load-misses/}:uS",
        "{cpu_core/instructions/,cpu_core/dTLB-stores/,cpu_core/dTLB-store-misses/,cpu_core/mem-stores/}:uS"
    ]

    # 8 E-Core Groups (Updated with exact casing, NO fp_arith_inst_retired)
    e_groups = [
        "{cpu_atom/instructions/,cpu_atom/cpu-cycles/,cpu_atom/bus-cycles/}:uS",
        "{cpu_atom/instructions/,cpu_atom/branch-loads/,cpu_atom/branch-load-misses/}:uS",
        "{cpu_atom/instructions/,cpu_atom/br_inst_retired.all_branches/,cpu_atom/br_misp_retired.all_branches/,cpu_atom/ref-cycles/}:uS",
        "{cpu_atom/instructions/,cpu_atom/L1-dcache-loads/,cpu_atom/L1-dcache-stores/}:uS",
        "{cpu_atom/instructions/,cpu_atom/L1-icache-load-misses/,cpu_atom/LLC-loads/,cpu_atom/LLC-load-misses/}:uS",
        "{cpu_atom/instructions/,cpu_atom/cache-references/,cpu_atom/cache-misses/,cpu_atom/mem-loads/}:uS",
        "{cpu_atom/instructions/,cpu_atom/dTLB-loads/,cpu_atom/dTLB-load-misses/,cpu_atom/iTLB-load-misses/}:uS",
        "{cpu_atom/instructions/,cpu_atom/dTLB-stores/,cpu_atom/dTLB-store-misses/,cpu_atom/mem-stores/}:uS"
    ]

    with open(output_file, "w") as f:
        f.write("#!/bin/bash\n\n")

        # --- DaCapo P-Core ---
        for benchmark in dacapo_benchmarks: 
            for i, group in enumerate(p_groups):
                for freq in frequencies:
                    for cpu in p_cpus:
                        flags_to_use = strict_flags if benchmark in strict_benchmarks else relaxed_flags
                        command = f"perf record -a -C {cpu} -e \"{group}\" -c 10000000 --no-buffering -o cpu_{cpu}_{freq}GHz_dacapo_{benchmark}_10000000_{i}_0.out -- "
                        command += f"taskset --cpu-list {cpu} java {flags_to_use} -jar {dacapo_dir}/dacapo-23.11-MR2-chopin.jar -s default -t 1 -n 1 {benchmark}"
                        f.write(command + "\n") 

        # --- DaCapo E-Core ---
        for i, group in enumerate(e_groups):
            for freq in frequencies:
                for cpu in e_cpus:
                    for benchmark in dacapo_benchmarks:
                        flags_to_use = strict_flags if benchmark in strict_benchmarks else relaxed_flags
                        command = f"perf record -a -C {cpu} -e \"{group}\" -c 10000000 --no-buffering -o cpu_{cpu}_{freq}GHz_dacapo_{benchmark}_10000000_{i}_0.out -- "
                        command += f"taskset --cpu-list {cpu} java {flags_to_use} -jar {dacapo_dir}/dacapo-23.11-MR2-chopin.jar -s default -t 1 -n 1 {benchmark}"
                        f.write(command + "\n") 
        
        # --- Email Notification ---
        f.write("\n# Send email notification when the collection finishes\n")
        f.write(f"echo \"The highly deterministic DaCapo perf collection script has finished running all P-core and E-core workloads.\" | mail -s \"[Perf] DaCapo Collection Complete\" {email_address}\n")

    os.chmod(output_file, os.stat(output_file).st_mode | stat.S_IEXEC)
    print(f"Generated {output_file}")

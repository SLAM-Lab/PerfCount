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

    frequencies = ["3.0"]
    cpus = [0]
    dacapo_dir = "../../../benchmarks/dacapo"

    # Strict Flags (Maximum Determinism for x86)
    strict_flags = (
        "-Xcomp -Xbatch -XX:+TieredCompilation -XX:TieredStopAtLevel=1 -XX:CICompilerCount=1 "
        "-XX:+UnlockExperimentalVMOptions -XX:+UseSerialGC "
        "-Xms16g -Xmx16g -XX:InitialCodeCacheSize=256m -XX:ReservedCodeCacheSize=256m "
        "-XX:+DisableExplicitGC -XX:-UseBiasedLocking -XX:-UsePerfData -XX:-UseTLAB "
        "-XX:+UnlockDiagnosticVMOptions -XX:GuaranteedSafepointInterval=0"
    )

    relaxed_flags = (
        "-Xcomp -Xbatch -XX:+TieredCompilation -XX:TieredStopAtLevel=1 -XX:CICompilerCount=1 "
        "-XX:+UnlockExperimentalVMOptions -XX:+UseSerialGC "
        "-Xms16g -Xmx16g -XX:InitialCodeCacheSize=256m -XX:ReservedCodeCacheSize=256m "
        "-XX:+DisableExplicitGC -XX:-UseBiasedLocking -XX:-UsePerfData "
        "-XX:+UnlockDiagnosticVMOptions -XX:GuaranteedSafepointInterval=0"
    )   
    # 11 Hardware Counter Groups matching x86_server.cfg (runs 0-10)
    x86_groups = [
        "{instructions,cpu-cycles,bus-cycles,fp_arith_inst_retired.scalar_single}:uS",
        "{instructions,branch-loads,branch-load-misses}:uS",
        "{instructions,br_inst_retired.all_branches,br_misp_retired.all_branches,ref-cycles}:uS",
        "{instructions,L1-dcache-loads,L1-dcache-load-misses,L1-dcache-stores}:uS",
        "{instructions,L1-icache-load-misses,LLC-loads,LLC-load-misses}:uS",
        "{instructions,cache-references,cache-misses,mem-loads}:uS",
        "{instructions,dTLB-loads,dTLB-load-misses,iTLB-load-misses}:uS",
        "{instructions,dTLB-stores,dTLB-store-misses,mem-stores}:uS",
        "{instructions,branch-instructions,branch-misses,node-loads}:uS",
        "{instructions,node-load-misses,slots,fp_arith_inst_retired.scalar_double}:uS",
        "{instructions,fp_arith_inst_retired.128b_packed_single,fp_arith_inst_retired.128b_packed_double,fp_arith_inst_retired.256b_packed_single}:uS",
    ]

    with open(output_file, "w") as f:
        f.write("#!/bin/bash\n\n")
        count = 0

        for counter_group in range(0, len(x86_groups)):
            for freq in frequencies:
                for cpu in cpus:
                    for benchmark in dacapo_benchmarks:

                        flags_to_use = strict_flags if benchmark in strict_benchmarks else relaxed_flags

                        command = f"perf record -a -C {cpu} "
                        command += f"-e \"{x86_groups[counter_group]}\" "
                        command += "-c 10000000 --no-buffering -o "
                        command += f"cpu_{cpu}_{freq}GHz_dacapo_{benchmark}_10000000_{counter_group}_0.out -- "
                        command += f"taskset --cpu-list {cpu} java {flags_to_use} -jar {dacapo_dir}/dacapo-23.11-MR2-chopin.jar -s default -t 1 -n 1 -f 5 {benchmark}"

                        f.write(command + "\n")
                        count += 1

        f.write(f"\necho \"DaCapo x86 server collection complete.\" | mail -s \"DaCapo Collection Complete\" {email_address}\n")

    os.chmod(output_file, os.stat(output_file).st_mode | stat.S_IEXEC)

    print(f"Generated {output_file} with {count} DaCapo commands "
          f"({len(dacapo_benchmarks)} benchmarks × {len(frequencies)} freq × "
          f"{len(cpus)} cpu × {len(x86_groups)} groups).")

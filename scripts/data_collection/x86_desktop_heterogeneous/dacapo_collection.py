#!/usr/bin/env python3
"""
Generate dacapo_collection.sh for the x86 desktop heterogeneous platform
(Intel i7-13700: P-core cpu0 + E-core cpu16).

Collects HW counter traces at 4 frequencies on both core types, plus
run 100 for simultaneous perf + RAPL power via spec_power_wrapper.sh.
"""

import os
import stat

if __name__ == "__main__":
    output_file = "dacapo_collection.sh"
    email_address = "mebarondeau@utexas.edu"

    strict_benchmarks = [
        'avrora', 'batik', 'biojava', 'eclipse', 'fop', 'graphchi', 'h2', 'jme',
        'luindex', 'lusearch', 'pmd', 'sunflow', 'xalan', 'zxing'
    ]

    relaxed_benchmarks = [
        'cassandra', 'h2o', 'jython', 'kafka', 'spring', 'tomcat', 'tradebeans', 'tradesoap'
    ]

    dacapo_benchmarks = strict_benchmarks + relaxed_benchmarks

    frequencies = ["4.0"]
    p_cpus = [0]
    e_cpus = [16]
    dacapo_dir = "../../../benchmarks/dacapo"
    power_wrapper = "/home/local/bumblebee/meb4744/PerfCount/scripts/data_collection/x86_desktop_heterogeneous/spec_power_wrapper.sh"
    output_root = "/home/local/bumblebee/meb4744/PerfCount"

    strict_flags = (
        "-Xcomp -Xbatch -XX:+TieredCompilation -XX:TieredStopAtLevel=1 "
        "-XX:CICompilerCount=1 "
        "-XX:+UnlockExperimentalVMOptions -XX:+UseSerialGC -XX:+AlwaysPreTouch "
        "-Xms16g -Xmx16g -XX:InitialCodeCacheSize=256m -XX:ReservedCodeCacheSize=256m "
        "-XX:+DisableExplicitGC -XX:-UseBiasedLocking -XX:-UsePerfData -XX:-UseTLAB "
        "-XX:+UnlockDiagnosticVMOptions -XX:GuaranteedSafepointInterval=0"
    )

    relaxed_flags = (
        "-Xcomp -Xbatch -XX:+TieredCompilation -XX:TieredStopAtLevel=1 "
        "-XX:CICompilerCount=1 "
        "-XX:+UnlockExperimentalVMOptions -XX:+UseSerialGC -XX:+AlwaysPreTouch "
        "-Xms16g -Xmx16g -XX:InitialCodeCacheSize=256m -XX:ReservedCodeCacheSize=256m "
        "-XX:+DisableExplicitGC -XX:-UseBiasedLocking -XX:-UsePerfData "
        "-XX:+UnlockDiagnosticVMOptions -XX:GuaranteedSafepointInterval=0"
    )

    # P-core (Golden Cove) HW counter groups, matching x86_desktop_pcore_2017.cfg runs 0-13
    pcore_groups = [
        "{cpu_core/instructions/,cpu_core/cpu-cycles/,cpu_core/bus-cycles/,cpu_core/fp_arith_inst_retired.scalar_single/}:uS",
        "{cpu_core/instructions/,cpu_core/branch-loads/,cpu_core/branch-load-misses/}:uS",
        "{cpu_core/instructions/,cpu_core/br_inst_retired.all_branches/,cpu_core/br_misp_retired.all_branches/,cpu_core/ref-cycles/}:uS",
        "{cpu_core/instructions/,cpu_core/L1-dcache-loads/,cpu_core/L1-dcache-load-misses/,cpu_core/L1-dcache-stores/}:uS",
        "{cpu_core/instructions/,cpu_core/L1-icache-load-misses/,cpu_core/LLC-loads/,cpu_core/LLC-load-misses/}:uS",
        "{cpu_core/instructions/,cpu_core/cache-references/,cpu_core/cache-misses/,cpu_core/mem-loads/}:uS",
        "{cpu_core/instructions/,cpu_core/dTLB-loads/,cpu_core/dTLB-load-misses/,cpu_core/iTLB-load-misses/}:uS",
        "{cpu_core/instructions/,cpu_core/dTLB-stores/,cpu_core/dTLB-store-misses/,cpu_core/mem-stores/}:uS",
        "{cpu_core/instructions/,cpu_core/LLC-stores/,cpu_core/LLC-store-misses/,cpu_core/mem-loads-aux/}:uS",
        "{cpu_core/instructions/,cpu_core/branch-instructions/,cpu_core/branch-misses/,cpu_core/node-loads/}:uS",
        "{cpu_core/instructions/,cpu_core/node-load-misses/,cpu_core/fp_arith_inst_retired.scalar_double/}:uS",
        "{cpu_core/instructions/,cpu_core/fp_arith_inst_retired.128b_packed_single/,cpu_core/fp_arith_inst_retired.128b_packed_double/,cpu_core/fp_arith_inst_retired.256b_packed_single/}:uS",
        "{cpu_core/instructions/,cpu_core/fp_arith_inst_retired.256b_packed_double/,cpu_core/idq.dsb_uops/,cpu_core/idq.mite_uops/}:uS",
        "{cpu_core/instructions/,cpu_core/idq_uops_not_delivered.core/,cpu_core/uops_executed.thread/}:uS",
    ]
    pcore_power_events = "{cpu_core/instructions/,cpu_core/cpu-cycles/,cpu_core/ref-cycles/}:uS"

    # E-core (Gracemont) HW counter groups, matching x86_desktop_ecore_2017.cfg runs 0-9
    ecore_groups = [
        "{cpu_atom/instructions/,cpu_atom/cpu-cycles/,cpu_atom/bus-cycles/}:uS",
        "{cpu_atom/instructions/,cpu_atom/branch-loads/,cpu_atom/branch-load-misses/}:uS",
        "{cpu_atom/instructions/,cpu_atom/br_inst_retired.all_branches/,cpu_atom/br_misp_retired.all_branches/,cpu_atom/ref-cycles/}:uS",
        "{cpu_atom/instructions/,cpu_atom/L1-dcache-loads/,cpu_atom/L1-dcache-load-misses/,cpu_atom/L1-dcache-stores/}:uS",
        "{cpu_atom/instructions/,cpu_atom/L1-icache-load-misses/,cpu_atom/LLC-loads/,cpu_atom/LLC-load-misses/}:uS",
        "{cpu_atom/instructions/,cpu_atom/cache-references/,cpu_atom/cache-misses/,cpu_atom/mem-loads/}:uS",
        "{cpu_atom/instructions/,cpu_atom/dTLB-loads/,cpu_atom/dTLB-load-misses/,cpu_atom/iTLB-load-misses/}:uS",
        "{cpu_atom/instructions/,cpu_atom/dTLB-stores/,cpu_atom/dTLB-store-misses/,cpu_atom/mem-stores/}:uS",
        "{cpu_atom/instructions/,cpu_atom/LLC-stores/,cpu_atom/LLC-store-misses/,cpu_atom/L1-icache-loads/}:uS",
        "{cpu_atom/instructions/,cpu_atom/branch-instructions/,cpu_atom/branch-misses/}:uS",
    ]
    ecore_power_events = "{cpu_atom/instructions/,cpu_atom/cpu-cycles/,cpu_atom/ref-cycles/}:uS"

    def java_cmd(cpu, benchmark):
        flags = strict_flags if benchmark in strict_benchmarks else relaxed_flags
        return (f"taskset --cpu-list {cpu} java {flags} "
                f"-jar {dacapo_dir}/dacapo-23.11-MR2-chopin.jar "
                f"-s default -t 1 -n 1 -f 5 {benchmark}")

    with open(output_file, "w") as f:
        f.write("#!/bin/bash\n\n")
        count = 0

        # ---- P-Core HW counter collection ----
        f.write("# === P-Core (cpu0) HW Counter Collection ===\n")
        for run_number in range(len(pcore_groups)):
            for freq in frequencies:
                for cpu in p_cpus:
                    for benchmark in dacapo_benchmarks:
                        base = f"{output_root}/cpu_{cpu}_{freq}GHz_dacapo_{benchmark}_10000000_{run_number}_0"
                        cmd = (f"perf record -a -C {cpu} "
                               f"-e \"{pcore_groups[run_number]}\" "
                               f"-c 10000000 --no-buffering -o {base}.out -- "
                               f"{java_cmd(cpu, benchmark)}")
                        f.write(cmd + "\n")
                        count += 1

        # ---- P-Core power collection (run 100) ----
        f.write("\n# === P-Core (cpu0) Power Collection (run 100) ===\n")
        for freq in frequencies:
            for cpu in p_cpus:
                f.write(f"perf stat -I 10 -x, -e power/energy-cores/ -o "
                        f"{output_root}/idle_{cpu}_{freq}GHz_power.csv "
                        f"sleep 10\n")
                count += 1
                for benchmark in dacapo_benchmarks:
                    base = f"{output_root}/cpu_{cpu}_{freq}GHz_dacapo_{benchmark}_10000000_100_0"
                    cmd = (f"{power_wrapper} {base} "
                           f"'{pcore_power_events}' "
                           f"{java_cmd(cpu, benchmark)}")
                    f.write(cmd + "\n")
                    count += 1

        # ---- E-Core HW counter collection ----
        f.write("\n# === E-Core (cpu16) HW Counter Collection ===\n")
        for run_number in range(len(ecore_groups)):
            for freq in frequencies:
                for cpu in e_cpus:
                    for benchmark in dacapo_benchmarks:
                        base = f"{output_root}/cpu_{cpu}_{freq}GHz_dacapo_{benchmark}_10000000_{run_number}_0"
                        cmd = (f"perf record -a -C {cpu} "
                               f"-e \"{ecore_groups[run_number]}\" "
                               f"-c 10000000 --no-buffering -o {base}.out -- "
                               f"{java_cmd(cpu, benchmark)}")
                        f.write(cmd + "\n")
                        count += 1

        # ---- E-Core power collection (run 100) ----
        f.write("\n# === E-Core (cpu16) Power Collection (run 100) ===\n")
        for freq in frequencies:
            for cpu in e_cpus:
                f.write(f"perf stat -I 10 -x, -e power/energy-cores/ -o "
                        f"{output_root}/idle_{cpu}_{freq}GHz_power.csv "
                        f"sleep 10\n")
                count += 1
                for benchmark in dacapo_benchmarks:
                    base = f"{output_root}/cpu_{cpu}_{freq}GHz_dacapo_{benchmark}_10000000_100_0"
                    cmd = (f"{power_wrapper} {base} "
                           f"'{ecore_power_events}' "
                           f"{java_cmd(cpu, benchmark)}")
                    f.write(cmd + "\n")
                    count += 1

        f.write(f"\necho \"DaCapo x86 desktop heterogeneous collection complete.\" | "
                f"mail -s \"DaCapo Collection Complete\" {email_address}\n")

    os.chmod(output_file, os.stat(output_file).st_mode | stat.S_IEXEC)

    n_bench = len(dacapo_benchmarks)
    n_freq = len(frequencies)
    n_pcore_runs = len(pcore_groups) + 1  # +1 for power
    n_ecore_runs = len(ecore_groups) + 1
    print(f"Generated {output_file} with {count} commands:")
    print(f"  P-core: {n_bench} benchmarks x {n_freq} freq x {len(pcore_groups)} HW groups + power = "
          f"{n_bench * n_freq * len(pcore_groups) + n_bench * n_freq + n_freq}")
    print(f"  E-core: {n_bench} benchmarks x {n_freq} freq x {len(ecore_groups)} HW groups + power = "
          f"{n_bench * n_freq * len(ecore_groups) + n_bench * n_freq + n_freq}")

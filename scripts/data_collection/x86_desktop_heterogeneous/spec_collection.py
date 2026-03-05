#!/usr/bin/env python3

import os
import stat

if __name__ == "__main__":
    output_file = "spec_collection.sh"

    spec_benchmarks = [
        500, 502, 505, 520, 523, 525, 531, 541, 548, 557,
        503, 507, 508, 510, 511, 519, 521, 526, 527, 538, 544, 549, 554
    ]
    
    frequencies = ["4.0"] # Adjust as needed
    p_cpus = [0]          # P-core
    e_cpus = [16]         # E-core
    spec_dir = "../../../benchmarks/spec"

    with open(output_file, "w") as f:
        f.write("#!/bin/bash\n\n")
        count = 0

        # --- SPEC P-Core Collection ---
        for run_number in range(0, 8):
            for freq in frequencies:
                for cpu in p_cpus:
                    for benchmark in spec_benchmarks:
                        command = "taskset --cpu-list " + str(cpu)
                        command += " bash -c 'cd " + spec_dir + " && source shrc && "
                        command += "runcpu --config=x86-desktop-heterogeneous-pcore "
                        command += "--define my_cpu=" + str(cpu) + " "
                        command += "--define my_freq=" + str(freq) + "GHz "
                        command += "--define my_run=" + str(run_number) + " "
                        command += str(benchmark) + "'" 
                        f.write(command + "\n") 
                        count += 1

        # --- SPEC E-Core Collection ---
        for run_number in range(0, 8):
            for freq in frequencies:
                for cpu in e_cpus:
                    for benchmark in spec_benchmarks:
                        command = "taskset --cpu-list " + str(cpu)
                        command += " bash -c 'cd " + spec_dir + " && source shrc && "
                        command += "runcpu --config=x86-desktop-heterogeneous-ecore "
                        command += "--define my_cpu=" + str(cpu) + " "
                        command += "--define my_freq=" + str(freq) + "GHz "
                        command += "--define my_run=" + str(run_number) + " "
                        command += str(benchmark) + "'" 
                        f.write(command + "\n") 
                        count += 1

        f.write("\necho \"SPEC x86 Data collection completed\" | mail -s \"SPEC Complete\" -r \"mebarondeau@utexas.edu\" \"mebarondeau@utexas.edu\"\n")

    st = os.stat(output_file)
    os.chmod(output_file, st.st_mode | stat.S_IEXEC)
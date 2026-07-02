#!/usr/bin/env python3

import os
import stat

if __name__ == "__main__":
    output_file = "spec_2017_collection.sh"

    spec_benchmarks = [
        500, 502, 505, 520, 523, 525, 531, 541, 548, 557,
        503, 507, 508, 510, 511, 519, 521, 527, 538, 544, 549, 554
    ]
    
    # Heterogeneous Edge Setup
    frequencies = ["1.0"]
    cpus = [4, 1]
    spec_dir = "../../../benchmarks/spec"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    power_sampler = os.path.join(script_dir, "arm_power_sampler.py")

    with open(output_file, "w") as f:
        f.write("#!/bin/bash\n\n")
        count = 0

        # --- SPEC 2017 Counter Collection (runs 0-24) ---
        for run_number in range(0, 25):
            for freq in frequencies:
                for cpu in cpus:
                    for benchmark in spec_benchmarks:
                        command = "taskset --cpu-list " + str(cpu)
                        command += " bash -c 'cd " + spec_dir + " && source shrc && "
                        command += "runcpu --config=arm_edge_heterogeneous "
                        command += "--define my_cpu=" + str(cpu) + " "
                        command += "--define my_freq=" + str(freq) + "GHz "
                        command += "--define my_run=" + str(run_number) + " "
                        command += str(benchmark) + "'"
                        f.write(command + "\n")
                        count += 1

        # --- SPEC 2017 Power Collection (run 100) ---
        for freq in frequencies:
            for cpu in cpus:
                # Idle power baseline
                idle_base = f"/home/meb4744/PerfCount/idle_{cpu}_{freq}GHz"
                f.write(f"python3 {power_sampler} {idle_base}_power.csv &\n")
                f.write("IDLE_PID=$!\n")
                f.write("sleep 10\n")
                f.write("kill $IDLE_PID 2>/dev/null; wait $IDLE_PID 2>/dev/null\n")
                count += 1

                for benchmark in spec_benchmarks:
                    command = "taskset --cpu-list " + str(cpu)
                    command += " bash -c 'cd " + spec_dir + " && source shrc && "
                    command += "runcpu --config=arm_edge_heterogeneous "
                    command += "--define my_cpu=" + str(cpu) + " "
                    command += "--define my_freq=" + str(freq) + "GHz "
                    command += "--define my_run=100 "
                    command += str(benchmark) + "'"
                    f.write(command + "\n")
                    count += 1


        # Email notification block
        f.write("\n")
        f.write("END_TIME=$(date)\n")
        f.write("HOSTNAME=$(hostname)\n")
        f.write("echo \"SPEC Edge Data collection completed at $END_TIME on $HOSTNAME\" | mail -s \"SPEC Collection Complete\" -r \"meb4744@cs.utexas.edu\" \"mebarondeau@utexas.edu\"\n")

    st = os.stat(output_file)
    os.chmod(output_file, st.st_mode | stat.S_IEXEC)
    
    print(f"Generated {output_file} with {count} SPEC commands (20 runs per benchmark).")


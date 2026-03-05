#!/usr/bin/env python3

import os
import stat

if __name__ == "__main__":
    output_file = "spec_collection.sh"

    # SPEC 2017 benchmarks
    spec_benchmarks = [
         500, 502, 505, 520, 523, 525, 531, 541, 548, 557,
         503, 507, 508, 510, 511, 519, 521, 526, 527, 538, 544, 549, 554
    ]
    
    frequencies = ["3.0"]
    cpus = [0]
    spec_dir = "../../../benchmarks/spec"

    with open(output_file, "w") as f:
        f.write("#!/bin/bash\n\n")
        count = 0

        # --- SPEC 2017 Collection (18 Groups) ---
        for run_number in range(0, 18):  
            for freq in frequencies:
                for cpu in cpus:
                    for benchmark in spec_benchmarks:
                        command = "taskset --cpu-list " + str(cpu)
                        command += " bash -c 'cd " + spec_dir + " && source shrc && "
                        command += "runcpu --config=arm_desktop "
                        command += "--define my_cpu=" + str(cpu) + " "
                        command += "--define my_freq=" + str(freq) + "GHz "
                        command += "--define my_run=" + str(run_number) + " "
                        command += str(benchmark) + "'"
                        f.write(command + "\n")
                        count += 1


        # Add email notification at the end of the script
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
        f.write("Script: arm_edge_heterogeneous/collection.sh\n")
        f.write("Working Directory: $(pwd)\n")
        f.write("\n")
        f.write("All SPEC benchmarks have been processed.\n")
        f.write("\"\n")
        f.write("\n")
        f.write("echo \"$EMAIL_BODY\" | mail -s \"$EMAIL_SUBJECT\" -r \"$EMAIL_SENDER\" \"$EMAIL_RECIPIENT\"\n")
        f.write("\n")
        f.write("echo \"Data collection completed at $END_TIME\"\n")

    st = os.stat(output_file)
    os.chmod(output_file, st.st_mode | stat.S_IEXEC)
    
    print(f"Generated {output_file} with {count} SPEC commands (18 runs per benchmark).")

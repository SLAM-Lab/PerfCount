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
                        command = f"cd {spec_dir} && source shrc && "
                        command += f"runcpu --base --config=arm_desktop --define my_cpu={cpu} --define my_freq={freq}GHz "
                        command += f"--define my_run={run_number} {benchmark}" 
                        f.write(command + "\n") 
                        count += 1

        # Email notification block
        f.write("\n")
        f.write("END_TIME=$(date)\n")
        f.write("HOSTNAME=$(hostname)\n")
        f.write("echo \"SPEC Data collection completed at $END_TIME on $HOSTNAME\" | mail -s \"SPEC Collection Complete\" -r \"mebarondeau@utexas.edu\" \"mebarondeau@utexas.edu\"\n")

    st = os.stat(output_file)
    os.chmod(output_file, st.st_mode | stat.S_IEXEC)
    
    print(f"Generated {output_file} with {count} SPEC commands (18 runs per benchmark).")

#!/usr/bin/env python3

import os
import stat

if __name__ == "__main__":
    output_file = "spec_2017_collection.sh"
    email_address = "mebarondeau@utexas.edu"

    # SPEC 2017 benchmarks
    spec_benchmarks = [
        500, 502, 505, 520, 523, 525, 531, 541, 548, 557,
        503, 507, 508, 510, 511, 519, 521, 526, 527, 538, 544, 549, 554
    ]

    frequencies = ["1.0", "2.0", "3.0"]
    cpus = [0]
    spec_dir = "../../../benchmarks/spec"

    with open(output_file, "w") as f:
        f.write("#!/bin/bash\n\n")
        count = 0

        for run_number in range(0, 13):
            for freq in frequencies:
                for cpu in cpus:
                    for benchmark in spec_benchmarks:
                        command = "taskset --cpu-list " + str(cpu)
                        command += " bash -c 'cd " + spec_dir + " && source shrc && "
                        command += "runcpu --config=x86_server "
                        command += "--tune=base "
                        command += "--define my_cpu=" + str(cpu) + " "
                        command += "--define my_freq=" + str(freq) + "GHz "
                        command += "--define my_run=" + str(run_number) + " "
                        command += str(benchmark) + "'"
                        f.write(command + "\n")
                        count += 1

        f.write(f"\necho \"SPEC 2017 x86 server collection complete.\" | mail -s \"SPEC Collection Complete\" {email_address}\n")

    os.chmod(output_file, os.stat(output_file).st_mode | stat.S_IEXEC)

    print(f"Generated {output_file} with {count} SPEC commands "
          f"({len(spec_benchmarks)} benchmarks × {len(frequencies)} freq × "
          f"{len(cpus)} cpu × 13 runs).")

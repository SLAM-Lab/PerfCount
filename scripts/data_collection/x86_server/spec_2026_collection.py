#!/usr/bin/env python3

import os
import stat

if __name__ == "__main__":
    output_file = "spec_2026_collection.sh"
    email_address = "mebarondeau@utexas.edu"

    # SPEC 2026 rate benchmarks
    intrate_benchmarks = [706, 707, 708, 710, 714, 721, 723, 727, 729, 734, 735, 750, 753, 777]
    fprate_benchmarks  = [709, 722, 731, 736, 737, 748, 749, 765, 766, 767, 772, 782]

    # SPEC 2026 speed benchmarks
    intspeed_benchmarks = [801, 807, 817, 821, 823, 827, 829, 834, 835, 838, 846, 853, 854]
    fpspeed_benchmarks  = [800, 803, 809, 811, 816, 820, 822, 849, 857, 865, 867, 872, 881]

    spec_benchmarks = intrate_benchmarks + fprate_benchmarks #+ intspeed_benchmarks + fpspeed_benchmarks

    frequencies = ["3.0"]
    cpus = [0]
    spec_dir = "../../../benchmarks/spec_2026"

    with open(output_file, "w") as f:
        f.write("#!/bin/bash\n\n")
        count = 0

        for run_number in range(0, 11):
            for freq in frequencies:
                for cpu in cpus:
                    for benchmark in spec_benchmarks:
                        command = "taskset --cpu-list " + str(cpu)
                        command += " bash -c 'cd " + spec_dir + " && source shrc && "
                        command += "runcpu --config=x86_server_2026 "
                        command += "--tune=base "
                        command += "--define my_cpu=" + str(cpu) + " "
                        command += "--define my_freq=" + str(freq) + "GHz "
                        command += "--define my_run=" + str(run_number) + " "
                        command += str(benchmark) + "'"
                        f.write(command + "\n")
                        count += 1

        f.write(f"\necho \"SPEC 2026 x86 server collection complete.\" | mail -s \"SPEC 2026 Collection Complete\" {email_address}\n")

    os.chmod(output_file, os.stat(output_file).st_mode | stat.S_IEXEC)

    n_rate  = len(intrate_benchmarks)  + len(fprate_benchmarks)
    n_speed = len(intspeed_benchmarks) + len(fpspeed_benchmarks)
    print(f"Generated {output_file} with {count} SPEC 2026 commands "
          f"({n_rate} rate + {n_speed} speed benchmarks × {len(frequencies)} freq × "
          f"{len(cpus)} cpu × 11 runs).")

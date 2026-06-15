#!/usr/bin/env python3

import os
import stat

if __name__ == "__main__":
    output_file = "spec_2026_collection.sh"

    # SPEC 2026 rate benchmarks
    intrate_benchmarks  = [706, 707, 708, 710, 714, 721, 723, 727, 729, 734, 735, 750, 753, 777]
    fprate_benchmarks   = [709, 722, 731, 736, 737, 748, 749, 765, 766, 767, 772, 782]

    # SPEC 2026 speed benchmarks
    intspeed_benchmarks = [801, 807, 817, 821, 823, 827, 829, 834, 835, 838, 846, 853, 854]
    fpspeed_benchmarks  = [800, 803, 809, 811, 816, 820, 822, 849, 857, 865, 867, 872, 881]

    spec_benchmarks = intrate_benchmarks + fprate_benchmarks #+ intspeed_benchmarks + fpspeed_benchmarks

    email_recipient = "mebarondeau@utexas.edu"

    frequencies = ["4.0"]
    p_cpus = [0]   # P-core
    e_cpus = [16]  # E-core
    spec_dir = "../../../benchmarks/spec_2026"

    with open(output_file, "w") as f:
        f.write("#!/bin/bash\n\n")
        count = 0

        # --- SPEC 2026 P-Core Collection (17 counter groups, runs 0-16, plus run 100 for power) ---
        for run_number in list(range(0, 17)) + [100]:
            for freq in frequencies:
                for cpu in p_cpus:
                    for benchmark in spec_benchmarks:
                        command = "taskset --cpu-list " + str(cpu)
                        command += " bash -c 'cd " + spec_dir + " && source shrc && "
                        command += "runcpu --config=x86_desktop_pcore_2026 "
                        command += "--tune=base "
                        command += "--define my_cpu=" + str(cpu) + " "
                        command += "--define my_freq=" + str(freq) + "GHz "
                        command += "--define my_run=" + str(run_number) + " "
                        command += str(benchmark) + "'"
                        f.write(command + "\n")
                        count += 1

        # --- SPEC 2026 E-Core Collection (12 counter groups, run 10 removed: topdown unsupported on Atom, plus run 100 for power) ---
        for run_number in [r for r in range(13) if r != 10] + [100]:
            for freq in frequencies:
                for cpu in e_cpus:
                    for benchmark in spec_benchmarks:
                        command = "taskset --cpu-list " + str(cpu)
                        command += " bash -c 'cd " + spec_dir + " && source shrc && "
                        command += "runcpu --config=x86_desktop_ecore_2026 "
                        command += "--tune=base "
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
        f.write(f"EMAIL_SENDER=\"{email_recipient}\"\n")
        f.write(f"EMAIL_RECIPIENT=\"{email_recipient}\"\n")
        f.write("EMAIL_SUBJECT=\"PerfCount Data Collection Completed on $HOSTNAME\"\n")
        f.write("EMAIL_BODY=\"The data collection workloads have completed successfully.\n")
        f.write("\n")
        f.write("Hostname: $HOSTNAME\n")
        f.write("End Time: $END_TIME\n")
        f.write("Script: x86_desktop_heterogeneous/spec_2026_collection.sh\n")
        f.write("Working Directory: $(pwd)\n")
        f.write("\n")
        f.write("All SPEC 2026 benchmarks have been processed (intrate, fprate, intspeed, fpspeed).\n")
        f.write("\"\n")
        f.write("\n")
        f.write("echo \"$EMAIL_BODY\" | mail -s \"$EMAIL_SUBJECT\" -r \"$EMAIL_SENDER\" \"$EMAIL_RECIPIENT\"\n")
        f.write("\n")
        f.write("echo \"Data collection completed at $END_TIME\"\n")

    st = os.stat(output_file)
    os.chmod(output_file, st.st_mode | stat.S_IEXEC)

    n_rate  = len(intrate_benchmarks)  + len(fprate_benchmarks)
    n_speed = len(intspeed_benchmarks) + len(fpspeed_benchmarks)


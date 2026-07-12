#!/usr/bin/env python3

import os
import stat

if __name__ == "__main__":
    output_file = "spec_2026_collection.sh"

    # SPEC 2026 rate benchmarks (matching scripts/data_collection/arm_server/spec_2026_collection.py)
    intrate_benchmarks  = [706, 707, 708, 710, 714, 721, 723, 727, 729, 734, 735, 750, 753, 777]
    fprate_benchmarks   = [709, 722, 731, 736, 737, 748, 749, 765, 766, 767, 772, 782]

    # SPEC 2026 speed benchmarks
    intspeed_benchmarks = [801, 807, 817, 821, 823, 827, 829, 834, 835, 838, 846, 853, 854]
    fpspeed_benchmarks  = [800, 803, 809, 811, 816, 820, 822, 849, 857, 865, 867, 872, 881]

    spec_benchmarks = intrate_benchmarks + fprate_benchmarks #+ intspeed_benchmarks + fpspeed_benchmarks

    # Full 25 single-metric PMU groups defined in arm_edge_heterogeneous_2026.cfg (my_run 0-24)
    full_runs = list(range(0, 25))

    # Reduced set matching DaCapo's arm_groups_reduced: baseline (instructions/cpu-cycles)
    # plus the top 4 counters by feature importance from prior arm_edge_heterogeneous
    # cross-freq/cross-proc experiments (stalled_cycles_backend, bus_cycles,
    # l3d_cache_refill, br_retired). The my_run=100 power pass below always runs
    # alongside whichever counter set is chosen here.
    reduced_runs = [0, 4, 6, 20, 23]

    run_numbers = reduced_runs  # EDIT: swap to full_runs for the full 25-group sweep

    # Heterogeneous Edge Setup
    frequencies = ["1.0"]
    cpus = [1]
    spec_dir = "../../../benchmarks/spec_2026"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    power_sampler = os.path.join(script_dir, "arm_power_sampler.py")

    with open(output_file, "w") as f:
        f.write("#!/bin/bash\n\n")
        count = 0

        # --- SPEC 2026 Counter Collection ---
        for run_number in run_numbers:
            for freq in frequencies:
                for cpu in cpus:
                    for benchmark in spec_benchmarks:
                        command = "taskset --cpu-list " + str(cpu)
                        command += " bash -c 'cd " + spec_dir + " && source shrc && "
                        command += "runcpu --config=arm_edge_heterogeneous_2026 "
                        command += "--tune=base "
                        command += "--define my_cpu=" + str(cpu) + " "
                        command += "--define my_freq=" + str(freq) + "GHz "
                        command += "--define my_run=" + str(run_number) + " "
                        command += str(benchmark) + "'"
                        f.write(command + "\n")
                        count += 1

        # --- SPEC 2026 Power Collection (run 100) ---
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
                    command += "runcpu --config=arm_edge_heterogeneous_2026 "
                    command += "--tune=base "
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
        f.write("echo \"SPEC 2026 Edge Data collection completed at $END_TIME on $HOSTNAME\" | mail -s \"SPEC 2026 Collection Complete\" -r \"meb4744@cs.utexas.edu\" \"mebarondeau@utexas.edu\"\n")

    st = os.stat(output_file)
    os.chmod(output_file, st.st_mode | stat.S_IEXEC)

    print(f"Generated {output_file} with {count} SPEC 2026 commands "
          f"({len(spec_benchmarks)} benchmarks x {len(frequencies)} freq x {len(cpus)} cpu x "
          f"{len(run_numbers)} counter runs, plus a run-100 power pass per cpu/freq).")

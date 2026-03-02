#!/usr/bin/env python3

import os
import time

if __name__ == "__main__":
    # SPEC 2017 benchmarks
    spec_benchmarks = [
        500, 502, 505, 520, 523, 525, 531, 541, 548, 557,
        503, 507, 508, 510, 511, 519, 521, 526, 527, 538, 544, 549, 554
    ]
      
    cpus = [0]      # P-core
    atom_cpus = [16] # E-core   
    granularities = [100000000]
    frequencies = ['3.0GHz'] 
    dacapo_dir = '../../../benchmarks/dacapo/'    
    spec_dir = '../../..//benchmarks/spec'
    
    # P-core SPEC collection
    for run_number in range(0, 1):
        for cpu in cpus:
             for benchmark in spec_benchmarks:
                command = "taskset --cpu-list " + str(cpu)
                command += " bash -c 'cd " + spec_dir + " && source shrc && "
                command += "runcpu --config=matthew-1cpu-x86-desktop-heterogeneous-pcore "
                command += "--define my_cpu=" + str(cpu) + " "
                command += "--define my_freq=" + str(freq) + " "
                command += "--define my_run=" + str(run_number) + " "
                command += str(benchmark) + "'" 
                print(command)

    # E-core SPEC collection 
    for run_number in range(0, 1):
        for cpu in atom_cpus:
            for benchmark in spec_benchmarks:
                command = "taskset --cpu-list " + str(cpu)
                command += " bash -c 'cd " + spec_dir + " && source shrc && "
                command += "runcpu --config=matthew-1cpu-x86-desktop-heterogeneous-ecore "
                command += "--define my_cpu=" + str(cpu) + " "
                command += "--define my_freq=" + str(freq) + " "
                command += "--define my_run=" + str(run_number) + " "
                command += str(benchmark) + "'"
                print(command)

    # Add email notification at the end of the script
    print("\n")
    print("# Send email notification upon completion")
    print("END_TIME=$(date)")
    print("HOSTNAME=$(hostname)")
    print("EMAIL_SENDER=\"mebarondeau@utexas.edu\"")
    print("EMAIL_RECIPIENT=\"mebarondeau@utexas.edu\"")
    print("EMAIL_SUBJECT=\"PerfCount Data Collection Completed on $HOSTNAME\"")
    print("EMAIL_BODY=\"The data collection workloads have completed successfully.")
    print("")
    print("Hostname: $HOSTNAME")
    print("End Time: $END_TIME")
    print("Script: x86_desktop_heterogeneous/collection.sh")
    print("Working Directory: $(pwd)")
    print("")
    print("All SPEC benchmarks have been processed.")
    print("\"")
    print("")
    print("echo \"$EMAIL_BODY\" | mail -s \"$EMAIL_SUBJECT\" -r \"$EMAIL_SENDER\" \"$EMAIL_RECIPIENT\"")
    print("")
    print("echo \"Data collection completed at $END_TIME\"")
    print("echo \"Email notification sent to $EMAIL_RECIPIENT\"")

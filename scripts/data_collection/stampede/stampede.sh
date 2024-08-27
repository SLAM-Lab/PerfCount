taskset --cpu-list 0 perf stat -I 10 -e instructions,cpu-cycles,br_misp_retired.all_branches,br_inst_retired.all_branches -o cpu_0_freq_3.00GHz_epoch_10_500_0 -x, runcpu --config=matthew-1cpu 500


#!/bin/bash
SAMPLE_PERIOD=10


#intspeed
taskset --cpu-list 9 perf stat -I $SAMPLE_PERIOD -e mem_load_retired.l2_miss,mem_load_retired.l3_miss,br_misp_retired.all_branches,fp_arith_inst_retired.scalar_double -o 600_pcore.csv -x, runcpu --config=matthew-1cpu 600
taskset --cpu-list 9 perf stat -I $SAMPLE_PERIOD -e mem_load_retired.l2_miss,mem_load_retired.l3_miss,br_misp_retired.all_branches,fp_arith_inst_retired.scalar_double -o 602_pcore.csv -x, runcpu --config=matthew-1cpu 602
taskset --cpu-list 9 perf stat -I $SAMPLE_PERIOD -e mem_load_retired.l2_miss,mem_load_retired.l3_miss,br_misp_retired.all_branches,fp_arith_inst_retired.scalar_double -o 605_pcore.csv -x, runcpu --config=matthew-1cpu 605
taskset --cpu-list 9 perf stat -I $SAMPLE_PERIOD -e mem_load_retired.l2_miss,mem_load_retired.l3_miss,br_misp_retired.all_branches,fp_arith_inst_retired.scalar_double -o 620_pcore.csv -x, runcpu --config=matthew-1cpu 620
taskset --cpu-list 9 perf stat -I $SAMPLE_PERIOD -e mem_load_retired.l2_miss,mem_load_retired.l3_miss,br_misp_retired.all_branches,fp_arith_inst_retired.scalar_double -o 623_pcore.csv -x, runcpu --config=matthew-1cpu 623
taskset --cpu-list 9 perf stat -I $SAMPLE_PERIOD -e mem_load_retired.l2_miss,mem_load_retired.l3_miss,br_misp_retired.all_branches,fp_arith_inst_retired.scalar_double -o 625_pcore.csv -x, runcpu --config=matthew-1cpu 625
taskset --cpu-list 9 perf stat -I $SAMPLE_PERIOD -e mem_load_retired.l2_miss,mem_load_retired.l3_miss,br_misp_retired.all_branches,fp_arith_inst_retired.scalar_double -o 631_pcore.csv -x, runcpu --config=matthew-1cpu 631
taskset --cpu-list 9 perf stat -I $SAMPLE_PERIOD -e mem_load_retired.l2_miss,mem_load_retired.l3_miss,br_misp_retired.all_branches,fp_arith_inst_retired.scalar_double -o 641_pcore.csv -x, runcpu --config=matthew-1cpu 641
taskset --cpu-list 9 perf stat -I $SAMPLE_PERIOD -e mem_load_retired.l2_miss,mem_load_retired.l3_miss,br_misp_retired.all_branches,fp_arith_inst_retired.scalar_double -o 648_pcore.csv -x, runcpu --config=matthew-1cpu 648
taskset --cpu-list 9 perf stat -I $SAMPLE_PERIOD -e mem_load_retired.l2_miss,mem_load_retired.l3_miss,br_misp_retired.all_branches,fp_arith_inst_retired.scalar_double -o 657_pcore.csv -x, runcpu --config=matthew-1cpu 657

#fpspeed
#taskset --cpu-list 9 perf stat -I 10 -e mem_load_retired.l2_miss,mem_load_retired.l3_miss,br_misp_retired.all_branches,fp_arith_inst_retired.scalar_double -o 603_pcore.csv -x, runcpu --config=matthew-1cpu 603
#taskset --cpu-list 9 perf stat -I 10 -e mem_load_retired.l2_miss,mem_load_retired.l3_miss,br_misp_retired.all_branches,fp_arith_inst_retired.scalar_double -o 607_pcore.csv -x, runcpu --config=matthew-1cpu 607
#taskset --cpu-list 9 perf stat -I 10 -e mem_load_retired.l2_miss,mem_load_retired.l3_miss,br_misp_retired.all_branches,fp_arith_inst_retired.scalar_double -o 619_pcore.csv -x, runcpu --config=matthew-1cpu 619
#taskset --cpu-list 9 perf stat -I 10 -e mem_load_retired.l2_miss,mem_load_retired.l3_miss,br_misp_retired.all_branches,fp_arith_inst_retired.scalar_double -o 621_pcore.csv -x, runcpu --config=matthew-1cpu 621
#taskset --cpu-list 9 perf stat -I 10 -e mem_load_retired.l2_miss,mem_load_retired.l3_miss,br_misp_retired.all_branches,fp_arith_inst_retired.scalar_double -o 627_pcore.csv -x, runcpu --config=matthew-1cpu 627
#taskset --cpu-list 9 perf stat -I 10 -e mem_load_retired.l2_miss,mem_load_retired.l3_miss,br_misp_retired.all_branches,fp_arith_inst_retired.scalar_double -o 628_pcore.csv -x, runcpu --config=matthew-1cpu 628
#taskset --cpu-list 9 perf stat -I 10 -e mem_load_retired.l2_miss,mem_load_retired.l3_miss,br_misp_retired.all_branches,fp_arith_inst_retired.scalar_double -o 638_pcore.csv -x, runcpu --config=matthew-1cpu 638
#taskset --cpu-list 9 perf stat -I 10 -e mem_load_retired.l2_miss,mem_load_retired.l3_miss,br_misp_retired.all_branches,fp_arith_inst_retired.scalar_double -o 644_pcore.csv -x, runcpu --config=matthew-1cpu 644
#taskset --cpu-list 9 perf stat -I 10 -e mem_load_retired.l2_miss,mem_load_retired.l3_miss,br_misp_retired.all_branches,fp_arith_inst_retired.scalar_double -o 649_pcore.csv -x, runcpu --config=matthew-1cpu 649
#taskset --cpu-list 9 perf stat -I 10 -e mem_load_retired.l2_miss,mem_load_retired.l3_miss,br_misp_retired.all_branches,fp_arith_inst_retired.scalar_double -o 654_pcore.csv -x, runcpu --config=matthew-1cpu 654


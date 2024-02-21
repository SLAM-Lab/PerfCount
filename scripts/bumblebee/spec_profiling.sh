#!/bin/bash
SAMPLE_PERIOD=10

#intspeed
taskset --cpu-list 9 perf stat -I $SAMPLE_PERIOD -e mem_load_retired.l2_miss,mem_load_retired.l3_miss,br_misp_retired.all_branches,fp_arith_inst_retired.scalar_double -o 600_pcore_1.csv -x, runcpu --config=matthew-1cpu 600
taskset --cpu-list 9 perf stat -I $SAMPLE_PERIOD -e mem_load_retired.l2_miss,mem_load_retired.l3_miss,br_misp_retired.all_branches,fp_arith_inst_retired.scalar_double -o 602_pcore_1.csv -x, runcpu --config=matthew-1cpu 602
taskset --cpu-list 9 perf stat -I $SAMPLE_PERIOD -e mem_load_retired.l2_miss,mem_load_retired.l3_miss,br_misp_retired.all_branches,fp_arith_inst_retired.scalar_double -o 605_pcore_1.csv -x, runcpu --config=matthew-1cpu 605
taskset --cpu-list 9 perf stat -I $SAMPLE_PERIOD -e mem_load_retired.l2_miss,mem_load_retired.l3_miss,br_misp_retired.all_branches,fp_arith_inst_retired.scalar_double -o 620_pcore_1.csv -x, runcpu --config=matthew-1cpu 620
taskset --cpu-list 9 perf stat -I $SAMPLE_PERIOD -e mem_load_retired.l2_miss,mem_load_retired.l3_miss,br_misp_retired.all_branches,fp_arith_inst_retired.scalar_double -o 623_pcore_1.csv -x, runcpu --config=matthew-1cpu 623
taskset --cpu-list 9 perf stat -I $SAMPLE_PERIOD -e mem_load_retired.l2_miss,mem_load_retired.l3_miss,br_misp_retired.all_branches,fp_arith_inst_retired.scalar_double -o 625_pcore_1.csv -x, runcpu --config=matthew-1cpu 625
taskset --cpu-list 9 perf stat -I $SAMPLE_PERIOD -e mem_load_retired.l2_miss,mem_load_retired.l3_miss,br_misp_retired.all_branches,fp_arith_inst_retired.scalar_double -o 631_pcore_1.csv -x, runcpu --config=matthew-1cpu 631
taskset --cpu-list 9 perf stat -I $SAMPLE_PERIOD -e mem_load_retired.l2_miss,mem_load_retired.l3_miss,br_misp_retired.all_branches,fp_arith_inst_retired.scalar_double -o 641_pcore_1.csv -x, runcpu --config=matthew-1cpu 641
taskset --cpu-list 9 perf stat -I $SAMPLE_PERIOD -e mem_load_retired.l2_miss,mem_load_retired.l3_miss,br_misp_retired.all_branches,fp_arith_inst_retired.scalar_double -o 648_pcore_1.csv -x, runcpu --config=matthew-1cpu 648
taskset --cpu-list 9 perf stat -I $SAMPLE_PERIOD -e mem_load_retired.l2_miss,mem_load_retired.l3_miss,br_misp_retired.all_branches,fp_arith_inst_retired.scalar_double -o 657_pcore_1.csv -x, runcpu --config=matthew-1cpu 657

taskset --cpu-list 9 perf stat -I $SAMPLE_PERIOD -e instructions,cpu-cycles,L1-dcache-loads,L1-icache-loads -o 600_pcore_2.csv -x, runcpu --config=matthew-1cpu 600
taskset --cpu-list 9 perf stat -I $SAMPLE_PERIOD -e instructions,cpu-cycles,L1-dcache-loads,L1-icache-loads -o 602_pcore_2.csv -x, runcpu --config=matthew-1cpu 602
taskset --cpu-list 9 perf stat -I $SAMPLE_PERIOD -e instructions,cpu-cycles,L1-dcache-loads,L1-icache-loads -o 605_pcore_2.csv -x, runcpu --config=matthew-1cpu 605
taskset --cpu-list 9 perf stat -I $SAMPLE_PERIOD -e instructions,cpu-cycles,L1-dcache-loads,L1-icache-loads -o 620_pcore_2.csv -x, runcpu --config=matthew-1cpu 620
taskset --cpu-list 9 perf stat -I $SAMPLE_PERIOD -e instructions,cpu-cycles,L1-dcache-loads,L1-icache-loads -o 623_pcore_2.csv -x, runcpu --config=matthew-1cpu 623
taskset --cpu-list 9 perf stat -I $SAMPLE_PERIOD -e instructions,cpu-cycles,L1-dcache-loads,L1-icache-loads -o 625_pcore_2.csv -x, runcpu --config=matthew-1cpu 625
taskset --cpu-list 9 perf stat -I $SAMPLE_PERIOD -e instructions,cpu-cycles,L1-dcache-loads,L1-icache-loads -o 631_pcore_2.csv -x, runcpu --config=matthew-1cpu 631
taskset --cpu-list 9 perf stat -I $SAMPLE_PERIOD -e instructions,cpu-cycles,L1-dcache-loads,L1-icache-loads -o 641_pcore_2.csv -x, runcpu --config=matthew-1cpu 641
taskset --cpu-list 9 perf stat -I $SAMPLE_PERIOD -e instructions,cpu-cycles,L1-dcache-loads,L1-icache-loads -o 648_pcore_2.csv -x, runcpu --config=matthew-1cpu 648
taskset --cpu-list 9 perf stat -I $SAMPLE_PERIOD -e instructions,cpu-cycles,L1-dcache-loads,L1-icache-loads -o 657_pcore_2.csv -x, runcpu --config=matthew-1cpu 657


#fpspeed
taskset --cpu-list 9 perf stat -I $SAMPLE_PERIOD -e mem_load_retired.l2_miss,mem_load_retired.l3_miss,br_misp_retired.all_branches,fp_arith_inst_retired.scalar_double -o 603_pcore_1.csv -x, runcpu --config=matthew-1cpu 603
taskset --cpu-list 9 perf stat -I $SAMPLE_PERIOD -e mem_load_retired.l2_miss,mem_load_retired.l3_miss,br_misp_retired.all_branches,fp_arith_inst_retired.scalar_double -o 607_pcore_1.csv -x, runcpu --config=matthew-1cpu 607
taskset --cpu-list 9 perf stat -I $SAMPLE_PERIOD -e mem_load_retired.l2_miss,mem_load_retired.l3_miss,br_misp_retired.all_branches,fp_arith_inst_retired.scalar_double -o 619_pcore_1.csv -x, runcpu --config=matthew-1cpu 619
taskset --cpu-list 9 perf stat -I $SAMPLE_PERIOD -e mem_load_retired.l2_miss,mem_load_retired.l3_miss,br_misp_retired.all_branches,fp_arith_inst_retired.scalar_double -o 621_pcore_1.csv -x, runcpu --config=matthew-1cpu 621
taskset --cpu-list 9 perf stat -I $SAMPLE_PERIOD -e mem_load_retired.l2_miss,mem_load_retired.l3_miss,br_misp_retired.all_branches,fp_arith_inst_retired.scalar_double -o 627_pcore_1.csv -x, runcpu --config=matthew-1cpu 627
taskset --cpu-list 9 perf stat -I $SAMPLE_PERIOD -e mem_load_retired.l2_miss,mem_load_retired.l3_miss,br_misp_retired.all_branches,fp_arith_inst_retired.scalar_double -o 628_pcore_1.csv -x, runcpu --config=matthew-1cpu 628
taskset --cpu-list 9 perf stat -I $SAMPLE_PERIOD -e mem_load_retired.l2_miss,mem_load_retired.l3_miss,br_misp_retired.all_branches,fp_arith_inst_retired.scalar_double -o 638_pcore_1.csv -x, runcpu --config=matthew-1cpu 638
taskset --cpu-list 9 perf stat -I $SAMPLE_PERIOD -e mem_load_retired.l2_miss,mem_load_retired.l3_miss,br_misp_retired.all_branches,fp_arith_inst_retired.scalar_double -o 644_pcore_1.csv -x, runcpu --config=matthew-1cpu 644
taskset --cpu-list 9 perf stat -I $SAMPLE_PERIOD -e mem_load_retired.l2_miss,mem_load_retired.l3_miss,br_misp_retired.all_branches,fp_arith_inst_retired.scalar_double -o 649_pcore_1.csv -x, runcpu --config=matthew-1cpu 649
taskset --cpu-list 9 perf stat -I $SAMPLE_PERIOD -e mem_load_retired.l2_miss,mem_load_retired.l3_miss,br_misp_retired.all_branches,fp_arith_inst_retired.scalar_double -o 654_pcore_1.csv -x, runcpu --config=matthew-1cpu 654

taskset --cpu-list 9 perf stat -I $SAMPLE_PERIOD -e instructions,cpu-cycles,L1-dcache-loads,L1-icache-loads -o 603_pcore_2.csv -x, runcpu --config=matthew-1cpu 603
taskset --cpu-list 9 perf stat -I $SAMPLE_PERIOD -e instructions,cpu-cycles,L1-dcache-loads,L1-icache-loads -o 607_pcore_2.csv -x, runcpu --config=matthew-1cpu 607
taskset --cpu-list 9 perf stat -I $SAMPLE_PERIOD -e instructions,cpu-cycles,L1-dcache-loads,L1-icache-loads -o 619_pcore_2.csv -x, runcpu --config=matthew-1cpu 619
taskset --cpu-list 9 perf stat -I $SAMPLE_PERIOD -e instructions,cpu-cycles,L1-dcache-loads,L1-icache-loads -o 621_pcore_2.csv -x, runcpu --config=matthew-1cpu 621
taskset --cpu-list 9 perf stat -I $SAMPLE_PERIOD -e instructions,cpu-cycles,L1-dcache-loads,L1-icache-loads -o 627_pcore_2.csv -x, runcpu --config=matthew-1cpu 627
taskset --cpu-list 9 perf stat -I $SAMPLE_PERIOD -e instructions,cpu-cycles,L1-dcache-loads,L1-icache-loads -o 628_pcore_2.csv -x, runcpu --config=matthew-1cpu 628
taskset --cpu-list 9 perf stat -I $SAMPLE_PERIOD -e instructions,cpu-cycles,L1-dcache-loads,L1-icache-loads -o 638_pcore_2.csv -x, runcpu --config=matthew-1cpu 638
taskset --cpu-list 9 perf stat -I $SAMPLE_PERIOD -e instructions,cpu-cycles,L1-dcache-loads,L1-icache-loads -o 644_pcore_2.csv -x, runcpu --config=matthew-1cpu 644
taskset --cpu-list 9 perf stat -I $SAMPLE_PERIOD -e instructions,cpu-cycles,L1-dcache-loads,L1-icache-loads -o 649_pcore_2.csv -x, runcpu --config=matthew-1cpu 649
taskset --cpu-list 9 perf stat -I $SAMPLE_PERIOD -e instructions,cpu-cycles,L1-dcache-loads,L1-icache-loads -o 654_pcore_2.csv -x, runcpu --config=matthew-1cpu 654

taskset --cpu-list 16 perf stat -I $SAMPLE_PERIOD -e mem_load_retired.l2_miss,mem_load_retired.l3_miss,br_misp_retired.all_branches,fp_arith_inst_retired.scalar_double -o 600_ecore_1.csv -x, runcpu --config=matthew-1cpu 600
taskset --cpu-list 16 perf stat -I $SAMPLE_PERIOD -e mem_load_retired.l2_miss,mem_load_retired.l3_miss,br_misp_retired.all_branches,fp_arith_inst_retired.scalar_double -o 602_ecore_1.csv -x, runcpu --config=matthew-1cpu 602
taskset --cpu-list 16 perf stat -I $SAMPLE_PERIOD -e mem_load_retired.l2_miss,mem_load_retired.l3_miss,br_misp_retired.all_branches,fp_arith_inst_retired.scalar_double -o 605_ecore_1.csv -x, runcpu --config=matthew-1cpu 605
taskset --cpu-list 16 perf stat -I $SAMPLE_PERIOD -e mem_load_retired.l2_miss,mem_load_retired.l3_miss,br_misp_retired.all_branches,fp_arith_inst_retired.scalar_double -o 620_ecore_1.csv -x, runcpu --config=matthew-1cpu 620
taskset --cpu-list 16 perf stat -I $SAMPLE_PERIOD -e mem_load_retired.l2_miss,mem_load_retired.l3_miss,br_misp_retired.all_branches,fp_arith_inst_retired.scalar_double -o 623_ecore_1.csv -x, runcpu --config=matthew-1cpu 623
taskset --cpu-list 16 perf stat -I $SAMPLE_PERIOD -e mem_load_retired.l2_miss,mem_load_retired.l3_miss,br_misp_retired.all_branches,fp_arith_inst_retired.scalar_double -o 625_ecore_1.csv -x, runcpu --config=matthew-1cpu 625
taskset --cpu-list 16 perf stat -I $SAMPLE_PERIOD -e mem_load_retired.l2_miss,mem_load_retired.l3_miss,br_misp_retired.all_branches,fp_arith_inst_retired.scalar_double -o 631_ecore_1.csv -x, runcpu --config=matthew-1cpu 631
taskset --cpu-list 16 perf stat -I $SAMPLE_PERIOD -e mem_load_retired.l2_miss,mem_load_retired.l3_miss,br_misp_retired.all_branches,fp_arith_inst_retired.scalar_double -o 641_ecore_1.csv -x, runcpu --config=matthew-1cpu 641
taskset --cpu-list 16 perf stat -I $SAMPLE_PERIOD -e mem_load_retired.l2_miss,mem_load_retired.l3_miss,br_misp_retired.all_branches,fp_arith_inst_retired.scalar_double -o 648_ecore_1.csv -x, runcpu --config=matthew-1cpu 648
taskset --cpu-list 16 perf stat -I $SAMPLE_PERIOD -e mem_load_retired.l2_miss,mem_load_retired.l3_miss,br_misp_retired.all_branches,fp_arith_inst_retired.scalar_double -o 657_ecore_1.csv -x, runcpu --config=matthew-1cpu 657

taskset --cpu-list 16 perf stat -I $SAMPLE_PERIOD -e instructions,cpu-cycles,L1-dcache-loads,L1-icache-loads -o 600_ecore_2.csv -x, runcpu --config=matthew-1cpu 600
taskset --cpu-list 16 perf stat -I $SAMPLE_PERIOD -e instructions,cpu-cycles,L1-dcache-loads,L1-icache-loads -o 602_ecore_2.csv -x, runcpu --config=matthew-1cpu 602
taskset --cpu-list 16 perf stat -I $SAMPLE_PERIOD -e instructions,cpu-cycles,L1-dcache-loads,L1-icache-loads -o 605_ecore_2.csv -x, runcpu --config=matthew-1cpu 605
taskset --cpu-list 16 perf stat -I $SAMPLE_PERIOD -e instructions,cpu-cycles,L1-dcache-loads,L1-icache-loads -o 620_ecore_2.csv -x, runcpu --config=matthew-1cpu 620
taskset --cpu-list 16 perf stat -I $SAMPLE_PERIOD -e instructions,cpu-cycles,L1-dcache-loads,L1-icache-loads -o 623_ecore_2.csv -x, runcpu --config=matthew-1cpu 623
taskset --cpu-list 16 perf stat -I $SAMPLE_PERIOD -e instructions,cpu-cycles,L1-dcache-loads,L1-icache-loads -o 625_ecore_2.csv -x, runcpu --config=matthew-1cpu 625
taskset --cpu-list 16 perf stat -I $SAMPLE_PERIOD -e instructions,cpu-cycles,L1-dcache-loads,L1-icache-loads -o 631_ecore_2.csv -x, runcpu --config=matthew-1cpu 631
taskset --cpu-list 16 perf stat -I $SAMPLE_PERIOD -e instructions,cpu-cycles,L1-dcache-loads,L1-icache-loads -o 641_ecore_2.csv -x, runcpu --config=matthew-1cpu 641
taskset --cpu-list 16 perf stat -I $SAMPLE_PERIOD -e instructions,cpu-cycles,L1-dcache-loads,L1-icache-loads -o 648_ecore_2.csv -x, runcpu --config=matthew-1cpu 648
taskset --cpu-list 16 perf stat -I $SAMPLE_PERIOD -e instructions,cpu-cycles,L1-dcache-loads,L1-icache-loads -o 657_ecore_2.csv -x, runcpu --config=matthew-1cpu 657


#fpspeed
taskset --cpu-list 16 perf stat -I $SAMPLE_PERIOD -e mem_load_retired.l2_miss,mem_load_retired.l3_miss,br_misp_retired.all_branches,fp_arith_inst_retired.scalar_double -o 603_ecore_1.csv -x, runcpu --config=matthew-1cpu 603
taskset --cpu-list 16 perf stat -I $SAMPLE_PERIOD -e mem_load_retired.l2_miss,mem_load_retired.l3_miss,br_misp_retired.all_branches,fp_arith_inst_retired.scalar_double -o 607_ecore_1.csv -x, runcpu --config=matthew-1cpu 607
taskset --cpu-list 16 perf stat -I $SAMPLE_PERIOD -e mem_load_retired.l2_miss,mem_load_retired.l3_miss,br_misp_retired.all_branches,fp_arith_inst_retired.scalar_double -o 619_ecore_1.csv -x, runcpu --config=matthew-1cpu 619
taskset --cpu-list 16 perf stat -I $SAMPLE_PERIOD -e mem_load_retired.l2_miss,mem_load_retired.l3_miss,br_misp_retired.all_branches,fp_arith_inst_retired.scalar_double -o 621_ecore_1.csv -x, runcpu --config=matthew-1cpu 621
taskset --cpu-list 16 perf stat -I $SAMPLE_PERIOD -e mem_load_retired.l2_miss,mem_load_retired.l3_miss,br_misp_retired.all_branches,fp_arith_inst_retired.scalar_double -o 627_ecore_1.csv -x, runcpu --config=matthew-1cpu 627
taskset --cpu-list 16 perf stat -I $SAMPLE_PERIOD -e mem_load_retired.l2_miss,mem_load_retired.l3_miss,br_misp_retired.all_branches,fp_arith_inst_retired.scalar_double -o 628_ecore_1.csv -x, runcpu --config=matthew-1cpu 628
taskset --cpu-list 16 perf stat -I $SAMPLE_PERIOD -e mem_load_retired.l2_miss,mem_load_retired.l3_miss,br_misp_retired.all_branches,fp_arith_inst_retired.scalar_double -o 638_ecore_1.csv -x, runcpu --config=matthew-1cpu 638
taskset --cpu-list 16 perf stat -I $SAMPLE_PERIOD -e mem_load_retired.l2_miss,mem_load_retired.l3_miss,br_misp_retired.all_branches,fp_arith_inst_retired.scalar_double -o 644_ecore_1.csv -x, runcpu --config=matthew-1cpu 644
taskset --cpu-list 16 perf stat -I $SAMPLE_PERIOD -e mem_load_retired.l2_miss,mem_load_retired.l3_miss,br_misp_retired.all_branches,fp_arith_inst_retired.scalar_double -o 649_ecore_1.csv -x, runcpu --config=matthew-1cpu 649
taskset --cpu-list 16 perf stat -I $SAMPLE_PERIOD -e mem_load_retired.l2_miss,mem_load_retired.l3_miss,br_misp_retired.all_branches,fp_arith_inst_retired.scalar_double -o 654_ecore_1.csv -x, runcpu --config=matthew-1cpu 654

taskset --cpu-list 16 perf stat -I $SAMPLE_PERIOD -e instructions,cpu-cycles,L1-dcache-loads,L1-icache-loads -o 603_ecore_2.csv -x, runcpu --config=matthew-1cpu 603
taskset --cpu-list 16 perf stat -I $SAMPLE_PERIOD -e instructions,cpu-cycles,L1-dcache-loads,L1-icache-loads -o 607_ecore_2.csv -x, runcpu --config=matthew-1cpu 607
taskset --cpu-list 16 perf stat -I $SAMPLE_PERIOD -e instructions,cpu-cycles,L1-dcache-loads,L1-icache-loads -o 619_ecore_2.csv -x, runcpu --config=matthew-1cpu 619
taskset --cpu-list 16 perf stat -I $SAMPLE_PERIOD -e instructions,cpu-cycles,L1-dcache-loads,L1-icache-loads -o 621_ecore_2.csv -x, runcpu --config=matthew-1cpu 621
taskset --cpu-list 16 perf stat -I $SAMPLE_PERIOD -e instructions,cpu-cycles,L1-dcache-loads,L1-icache-loads -o 627_ecore_2.csv -x, runcpu --config=matthew-1cpu 627
taskset --cpu-list 16 perf stat -I $SAMPLE_PERIOD -e instructions,cpu-cycles,L1-dcache-loads,L1-icache-loads -o 628_ecore_2.csv -x, runcpu --config=matthew-1cpu 628
taskset --cpu-list 16 perf stat -I $SAMPLE_PERIOD -e instructions,cpu-cycles,L1-dcache-loads,L1-icache-loads -o 638_ecore_2.csv -x, runcpu --config=matthew-1cpu 638
taskset --cpu-list 16 perf stat -I $SAMPLE_PERIOD -e instructions,cpu-cycles,L1-dcache-loads,L1-icache-loads -o 644_ecore_2.csv -x, runcpu --config=matthew-1cpu 644
taskset --cpu-list 16 perf stat -I $SAMPLE_PERIOD -e instructions,cpu-cycles,L1-dcache-loads,L1-icache-loads -o 649_ecore_2.csv -x, runcpu --config=matthew-1cpu 649
taskset --cpu-list 16 perf stat -I $SAMPLE_PERIOD -e instructions,cpu-cycles,L1-dcache-loads,L1-icache-loads -o 654_ecore_2.csv -x, runcpu --config=matthew-1cpu 654


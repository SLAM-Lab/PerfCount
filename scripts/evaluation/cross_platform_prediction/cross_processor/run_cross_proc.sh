# E-Core to P-Core
#python3 cross_proc_x86.py --data_dir ../../../../processed_data_10M/x86_desktop_heterogeneous/ --out_dir ../../../../results/cross_platform/cross_processor/x86_desktop_heterogeneous_ecore_src_10M_all/ --strict_loocv --src_cpu 16 --tgt_cpu 0 --suite all
#python3 cross_proc_x86.py --data_dir ../../../../processed_data_100M/x86_desktop_heterogeneous/ --out_dir ../../../../results/cross_platform/cross_processor/x86_desktop_heterogeneous_ecore_src_100M_all/ --strict_loocv --src_cpu 16 --tgt_cpu 0 --suite all
#python3 cross_proc_x86.py --data_dir ../../../../processed_data_1000M/x86_desktop_heterogeneous/ --out_dir ../../../../results/cross_platform/cross_processor/x86_desktop_heterogeneous_ecore_src_1000_all/ --strict_loocv --src_cpu 16 --tgt_cpu 0 --suite all

#python3 cross_proc_x86.py --data_dir ../../../../processed_data_10M/x86_desktop_heterogeneous/ --out_dir ../../../../results/cross_platform/cross_processor/x86_desktop_heterogeneous_ecore_src_10M_spec/ --strict_loocv --src_cpu 16 --tgt_cpu 0 --suite spec
#python3 cross_proc_x86.py --data_dir ../../../../processed_data_100M/x86_desktop_heterogeneous/ --out_dir ../../../../results/cross_platform/cross_processor/x86_desktop_heterogeneous_ecore_src_100M_spec/ --strict_loocv --src_cpu 16 --tgt_cpu 0 --suite spec
#python3 cross_proc_x86.py --data_dir ../../../../processed_data_1000M/x86_desktop_heterogeneous/ --out_dir ../../../../results/cross_platform/cross_processor/x86_desktop_heterogeneous_ecore_src_1000M_spec/ --strict_loocv --src_cpu 16 --tgt_cpu 0 --suite spec

#python3 cross_proc_x86.py --data_dir ../../../../processed_data_10M/x86_desktop_heterogeneous/ --out_dir ../../../../results/cross_platform/cross_processor/x86_desktop_heterogeneous_ecore_src_10M_dacapo/ --strict_loocv --src_cpu 16 --tgt_cpu 0 --suite dacapo
#python3 cross_proc_x86.py --data_dir ../../../../processed_data_100M/x86_desktop_heterogeneous/ --out_dir ../../../../results/cross_platform/cross_processor/x86_desktop_heterogeneous_ecore_src_100M_dacapo/ --strict_loocv --src_cpu 16 --tgt_cpu 0 --suite dacapo
#python3 cross_proc_x86.py --data_dir ../../../../processed_data_1000M/x86_desktop_heterogeneous/ --out_dir ../../../../results/cross_platform/cross_processor/x86_desktop_heterogeneous_ecore_src_1000M_dacapo/ --strict_loocv --src_cpu 16 --tgt_cpu 0 --suite dacapo

# P-Core to E-Core
#python3 cross_proc_x86.py --data_dir ../../../../processed_data_10M/x86_desktop_heterogeneous/ --out_dir ../../../../results/cross_platform/cross_processor/x86_desktop_heterogeneous_pcore_src_10M_all/ --strict_loocv --src_cpu 0 --tgt_cpu 16 --suite all
#python3 cross_proc_x86.py --data_dir ../../../../processed_data_100M/x86_desktop_heterogeneous/ --out_dir ../../../../results/cross_platform/cross_processor/x86_desktop_heterogeneous_pcore_100M_src_all/ --strict_loocv --src_cpu 0 --tgt_cpu 16 --suite all
#python3 cross_proc_x86.py --data_dir ../../../../processed_data_1000M/x86_desktop_heterogeneous/ --out_dir ../../../../results/cross_platform/cross_processor/x86_desktop_heterogeneous_pcore_1000M_src_all/ --strict_loocv --src_cpu 0 --tgt_cpu 16 --suite all

#python3 cross_proc_x86.py --data_dir ../../../../processed_data_10M/x86_desktop_heterogeneous/ --out_dir ../../../../results/cross_platform/cross_processor/x86_desktop_heterogeneous_pcore_src_10M_spec/ --strict_loocv --src_cpu 0 --tgt_cpu 16 --suite spec
#python3 cross_proc_x86.py --data_dir ../../../../processed_data_100M/x86_desktop_heterogeneous/ --out_dir ../../../../results/cross_platform/cross_processor/x86_desktop_heterogeneous_pcore_src_100M_spec/ --strict_loocv --src_cpu 0 --tgt_cpu 16 --suite spec
#python3 cross_proc_x86.py --data_dir ../../../../processed_data_1000M/x86_desktop_heterogeneous/ --out_dir ../../../../results/cross_platform/cross_processor/x86_desktop_heterogeneous_pcore_src_1000M_spec/ --strict_loocv --src_cpu 0 --tgt_cpu 16 --suite spec

#python3 cross_proc_x86.py --data_dir ../../../../processed_data_10M/x86_desktop_heterogeneous/ --out_dir ../../../../results/cross_platform/cross_processor/x86_desktop_heterogeneous_pcore_src_10M_dacapo/ --strict_loocv --src_cpu 0 --tgt_cpu 16 --suite dacapo
#python3 cross_proc_x86.py --data_dir ../../../../processed_data_100M/x86_desktop_heterogeneous/ --out_dir ../../../../results/cross_platform/cross_processor/x86_desktop_heterogeneous_pcore_src_100M_dacapo/ --strict_loocv --src_cpu 0 --tgt_cpu 16 --suite dacapo
# python3 cross_proc_x86.py --data_dir ../../../../processed_data_1000M/x86_desktop_heterogeneous/ --out_dir ../../../../results/cross_platform/cross_processor/x86_desktop_heterogeneous_pcore_src_1000M_dacapo/ --strict_loocv --src_cpu 0 --tgt_cpu 16 --suite dacapo

# Arm InO to Arm OOO
# python3 cross_proc_arm.py --data_dir ../../../../processed_data_10M/arm_edge_heterogeneous/ --out_dir ../../../../results/cross_platform/cross_processor/arm_edge_heterogeneous_ooo_src_all/ --strict_loocv --src_cpu 4 --tgt_cpu 1 --suite all
# python3 cross_proc_arm.py --data_dir ../../../../processed_data_100M/arm_edge_heterogeneous/ --out_dir ../../../../results/cross_platform/cross_processor/arm_edge_heterogeneous_ooo_src_all/ --strict_loocv --src_cpu 4 --tgt_cpu 1 --suite all
# python3 cross_proc_arm.py --data_dir ../../../../processed_data_1000M/arm_edge_heterogeneous/ --out_dir ../../../../results/cross_platform/cross_processor/arm_edge_heterogeneous_ooo_src_all/ --strict_loocv --src_cpu 4 --tgt_cpu 1 --suite all

python3 cross_proc_arm.py --data_dir ../../../../processed_data_10M/arm_edge_heterogeneous/ --out_dir ../../../../results/cross_platform/cross_processor/arm_edge_heterogeneous_ooo_src_10M_spec/ --strict_loocv --src_cpu 4 --tgt_cpu 1 --suite spec
python3 cross_proc_arm.py --data_dir ../../../../processed_data_100M/arm_edge_heterogeneous/ --out_dir ../../../../results/cross_platform/cross_processor/arm_edge_heterogeneous_ooo_src_100M_spec/ --strict_loocv --src_cpu 4 --tgt_cpu 1 --suite spec
python3 cross_proc_arm.py --data_dir ../../../../processed_data_1000M/arm_edge_heterogeneous/ --out_dir ../../../../results/cross_platform/cross_processor/arm_edge_heterogeneous_ooo_src_1000M_spec/ --strict_loocv --src_cpu 4 --tgt_cpu 1 --suite spec

# python3 cross_proc_arm.py --data_dir ../../../../processed_data_10M/arm_edge_heterogeneous/ --out_dir ../../../../results/cross_platform/cross_processor/arm_edge_heterogeneous_ooo_src_dacapo/ --strict_loocv --src_cpu 4 --tgt_cpu 1 --suite dacapo
# python3 cross_proc_arm.py --data_dir ../../../../processed_data_100M/arm_edge_heterogeneous/ --out_dir ../../../../results/cross_platform/cross_processor/arm_edge_heterogeneous_ooo_src_dacapo/ --strict_loocv --src_cpu 4 --tgt_cpu 1 --suite dacapo
# python3 cross_proc_arm.py --data_dir ../../../../processed_data_1000M/arm_edge_heterogeneous/ --out_dir ../../../../results/cross_platform/cross_processor/arm_edge_heterogeneous_ooo_src_dacapo/ --strict_loocv --src_cpu 4 --tgt_cpu 1 --suite dacapo

# Arm OOO to Arm InO
# python3 cross_proc_arm.py --data_dir ../../../../processed_data_10M/arm_edge_heterogeneous/ --out_dir ../../../../results/cross_platform/cross_processor/arm_edge_heterogeneous_ino_src_all/ --strict_loocv --src_cpu 1 --tgt_cpu 4 --suite all
# python3 cross_proc_arm.py --data_dir ../../../../processed_data_100M/arm_edge_heterogeneous/ --out_dir ../../../../results/cross_platform/cross_processor/arm_edge_heterogeneous_ino_src_all/ --strict_loocv --src_cpu 1 --tgt_cpu 4 --suite all
# python3 cross_proc_arm.py --data_dir ../../../../processed_data_1000M/arm_edge_heterogeneous/ --out_dir ../../../../results/cross_platform/cross_processor/arm_edge_heterogeneous_ino_src_all/ --strict_loocv --src_cpu 1 --tgt_cpu 4 --suite all

python3 cross_proc_arm.py --data_dir ../../../../processed_data_10M/arm_edge_heterogeneous/ --out_dir ../../../../results/cross_platform/cross_processor/arm_edge_heterogeneous_ino_src_10M_spec/ --strict_loocv --src_cpu 1 --tgt_cpu 4 --suite spec
python3 cross_proc_arm.py --data_dir ../../../../processed_data_100M/arm_edge_heterogeneous/ --out_dir ../../../../results/cross_platform/cross_processor/arm_edge_heterogeneous_ino_src_100M_spec/ --strict_loocv --src_cpu 1 --tgt_cpu 4 --suite spec
python3 cross_proc_arm.py --data_dir ../../../../processed_data_1000M/arm_edge_heterogeneous/ --out_dir ../../../../results/cross_platform/cross_processor/arm_edge_heterogeneous_ino_src_1000M_spec/ --strict_loocv --src_cpu 1 --tgt_cpu 4 --suite spec

# python3 cross_proc_arm.py --data_dir ../../../../processed_data_10M/arm_edge_heterogeneous/ --out_dir ../../../../results/cross_platform/cross_processor/arm_edge_heterogeneous_ino_src_dacapo/ --strict_loocv --src_cpu 1 --tgt_cpu 4 --suite dacapo
# python3 cross_proc_arm.py --data_dir ../../../../processed_data_100M/arm_edge_heterogeneous/ --out_dir ../../../../results/cross_platform/cross_processor/arm_edge_heterogeneous_ino_src_dacapo/ --strict_loocv --src_cpu 1 --tgt_cpu 4 --suite dacapo
# python3 cross_proc_arm.py --data_dir ../../../../processed_data_1000M/arm_edge_heterogeneous/ --out_dir ../../../../results/cross_platform/cross_processor/arm_edge_heterogeneous_ino_src_dacapo/ --strict_loocv --src_cpu 1 --tgt_cpu 4 --suite dacapo

# All Workloads
python3 cross_freq_arm.py --data_dir ../../../../processed_data_10M/arm_server/ --out_dir ../../../../results/cross_platform/cross_frequency/arm_server_10M_all_noc2_unequal/ --strict_loocv --suite all 
# python3 cross_freq_arm.py --data_dir ../../../../processed_data_100M/arm_server/ --out_dir ../../../../results/cross_platform/cross_frequency/arm_server_100M_all/ --strict_loocv --suite all
# python3 cross_freq_arm.py --data_dir ../../../../processed_data_1000M/arm_server/ --out_dir ../../../../results/cross_platform/cross_frequency/arm_server_1000M_all/ --strict_loocv --suite all

# python3 cross_freq_arm.py --data_dir ../../../../processed_data_10M/arm_desktop/ --out_dir ../../../../results/cross_platform/cross_frequency/arm_desktop_10M_all/ --strict_loocv --suite all
# python3 cross_freq_arm.py --data_dir ../../../../processed_data_100M/arm_desktop/ --out_dir ../../../../results/cross_platform/cross_frequency/arm_desktop_100M_all/ --strict_loocv --suite all
# python3 cross_freq_arm.py --data_dir ../../../../processed_data_1000M/arm_desktop/ --out_dir ../../../../results/cross_platform/cross_frequency/arm_desktop_1000M_all/ --strict_loocv --suite all

# python3 cross_freq_x86.py --data_dir ../../../../processed_data_10M/x86_desktop_heterogeneous/ --out_dir ../../../../results/cross_platform/cross_frequency/x86_desktop_heterogeneous_pcore_10M_all/ --strict_loocv --target_cpu 0 --suite all
# python3 cross_freq_x86.py --data_dir ../../../../processed_data_100M/x86_desktop_heterogeneous/ --out_dir ../../../../results/cross_platform/cross_frequency/x86_desktop_heterogeneous_pcore_100M_all/ --strict_loocv --target_cpu 0 --suite all
# python3 cross_freq_x86.py --data_dir ../../../../processed_data_1000M/x86_desktop_heterogeneous/ --out_dir ../../../../results/cross_platform/cross_frequency/x86_desktop_heterogeneous_pcore_1000M_all/ --strict_loocv --target_cpu 0 --suite all

# python3 cross_freq_x86.py --data_dir ../../../../processed_data_10M/x86_desktop_heterogeneous/ --out_dir ../../../../results/cross_platform/cross_frequency/x86_desktop_heterogeneous_ecore_10M_all/ --strict_loocv --target_cpu 16 --suite all
# python3 cross_freq_x86.py --data_dir ../../../../processed_data_100M/x86_desktop_heterogeneous/ --out_dir ../../../../results/cross_platform/cross_frequency/x86_desktop_heterogeneous_ecore_100M_all/ --strict_loocv --target_cpu 16 --suite all
# python3 cross_freq_x86.py --data_dir ../../../../processed_data_1000M/x86_desktop_heterogeneous/ --out_dir ../../../../results/cross_platform/cross_frequency/x86_desktop_heterogeneous_ecore_1000M_all/ --strict_loocv --target_cpu 16 --suite all

# # SPEC Only
# python3 cross_freq_arm.py --data_dir ../../../../processed_data_10M/arm_server/ --out_dir ../../../../results/cross_platform/cross_frequency/arm_server_10M_spec/ --strict_loocv --suite spec
# python3 cross_freq_arm.py --data_dir ../../../../processed_data_100M/arm_server/ --out_dir ../../../../results/cross_platform/cross_frequency/arm_server_100M_spec/ --strict_loocv --suite spec
# python3 cross_freq_arm.py --data_dir ../../../../processed_data_1000M/arm_server/ --out_dir ../../../../results/cross_platform/cross_frequency/arm_server_1000M_spec/ --strict_loocv --suite spec

# python3 cross_freq_arm.py --data_dir ../../../../processed_data_10M/arm_desktop/ --out_dir ../../../../results/cross_platform/cross_frequency/arm_desktop_10M_spec/ --strict_loocv --suite spec
# python3 cross_freq_arm.py --data_dir ../../../../processed_data_100M/arm_desktop/ --out_dir ../../../../results/cross_platform/cross_frequency/arm_desktop_100M_spec/ --strict_loocv --suite spec
# python3 cross_freq_arm.py --data_dir ../../../../processed_data_1000M/arm_desktop/ --out_dir ../../../../results/cross_platform/cross_frequency/arm_desktop_1000M_spec/ --strict_loocv --suite spec

# python3 cross_freq_x86.py --data_dir ../../../../processed_data_10M/x86_desktop_heterogeneous/ --out_dir ../../../../results/cross_platform/cross_frequency/x86_desktop_heterogeneous_pcore_10M_spec/ --strict_loocv --target_cpu 0 --suite spec
# python3 cross_freq_x86.py --data_dir ../../../../processed_data_100M/x86_desktop_heterogeneous/ --out_dir ../../../../results/cross_platform/cross_frequency/x86_desktop_heterogeneous_pcore_100M_spec/ --strict_loocv --target_cpu 0 --suite spec
# python3 cross_freq_x86.py --data_dir ../../../../processed_data_1000M/x86_desktop_heterogeneous/ --out_dir ../../../../results/cross_platform/cross_frequency/x86_desktop_heterogeneous_pcore_1000M_spec/ --strict_loocv --target_cpu 0 --suite spec

# python3 cross_freq_x86.py --data_dir ../../../../processed_data_10M/x86_desktop_heterogeneous/ --out_dir ../../../../results/cross_platform/cross_frequency/x86_desktop_heterogeneous_ecore_10M_spec/ --strict_loocv --target_cpu 16 --suite spec
# python3 cross_freq_x86.py --data_dir ../../../../processed_data_100M/x86_desktop_heterogeneous/ --out_dir ../../../../results/cross_platform/cross_frequency/x86_desktop_heterogeneous_ecore_100M_spec/ --strict_loocv --target_cpu 16 --suite spec
# python3 cross_freq_x86.py --data_dir ../../../../processed_data_1000M/x86_desktop_heterogeneous/ --out_dir ../../../../results/cross_platform/cross_frequency/x86_desktop_heterogeneous_ecore_1000M_spec/ --strict_loocv --target_cpu 16 --suite spec

# # DACAPO Only
# python3 cross_freq_arm.py --data_dir ../../../../processed_data_10M/arm_server/ --out_dir ../../../../results/cross_platform/cross_frequency/arm_server_10M_dacapo_noc2/ --strict_loocv --suite dacapo
# python3 cross_freq_arm.py --data_dir ../../../../processed_data_100M/arm_server/ --out_dir ../../../../results/cross_platform/cross_frequency/arm_server_100M_dacapo/ --strict_loocv --suite dacapo
# python3 cross_freq_arm.py --data_dir ../../../../processed_data_1000M/arm_server/ --out_dir ../../../../results/cross_platform/cross_frequency/arm_server_1000M_dacapo/ --strict_loocv --suite dacapo

# python3 cross_freq_arm.py --data_dir ../../../../processed_data_10M/arm_desktop/ --out_dir ../../../../results/cross_platform/cross_frequency/arm_desktop_10M_dacapo/ --strict_loocv --suite dacapo
# python3 cross_freq_arm.py --data_dir ../../../../processed_data_100M/arm_desktop/ --out_dir ../../../../results/cross_platform/cross_frequency/arm_desktop_100M_dacapo/ --strict_loocv --suite dacapo
# python3 cross_freq_arm.py --data_dir ../../../../processed_data_1000M/arm_desktop/ --out_dir ../../../../results/cross_platform/cross_frequency/arm_desktop_1000M_dacapo/ --strict_loocv --suite dacapo

# python3 cross_freq_x86.py --data_dir ../../../../processed_data_10M/x86_desktop_heterogeneous/ --out_dir ../../../../results/cross_platform/cross_frequency/x86_desktop_heterogeneous_pcore_10M_dacapo/ --strict_loocv --target_cpu 0 --suite dacapo
# python3 cross_freq_x86.py --data_dir ../../../../processed_data_100M/x86_desktop_heterogeneous/ --out_dir ../../../../results/cross_platform/cross_frequency/x86_desktop_heterogeneous_pcore_100M_dacapo/ --strict_loocv --target_cpu 0 --suite dacapo
# python3 cross_freq_x86.py --data_dir ../../../../processed_data_1000M/x86_desktop_heterogeneous/ --out_dir ../../../../results/cross_platform/cross_frequency/x86_desktop_heterogeneous_pcore_1000M_dacapo/ --strict_loocv --target_cpu 0 --suite dacapo

# python3 cross_freq_x86.py --data_dir ../../../../processed_data_10M/x86_desktop_heterogeneous/ --out_dir ../../../../results/cross_platform/cross_frequency/x86_desktop_heterogeneous_ecore_10M_dacapo/ --strict_loocv --target_cpu 16 --suite dacapo
# python3 cross_freq_x86.py --data_dir ../../../../processed_data_100M/x86_desktop_heterogeneous/ --out_dir ../../../../results/cross_platform/cross_frequency/x86_desktop_heterogeneous_ecore_100M_dacapo/ --strict_loocv --target_cpu 16 --suite dacapo
# python3 cross_freq_x86.py --data_dir ../../../../processed_data_1000M/x86_desktop_heterogeneous/ --out_dir ../../../../results/cross_platform/cross_frequency/x86_desktop_heterogeneous_ecore_1000M_dacapo/ --strict_loocv --target_cpu 16 --suite dacapo


#!/bin/bash
# Cross-system ARM prediction: arm_server <-> arm_desktop <-> arm_edge_heterogeneous
# Uses cross_sys_arm.py with --strict_loocv and --suite {spec,dacapo,all}
#
# Platform labels (defaults to basename + cpu suffix):
#   arm_server                        -- no cpu filter needed
#   arm_desktop                       -- no cpu filter needed
#   arm_edge_heterogeneous --tgt_cpu 4  (Out-of-Order core)
#   arm_edge_heterogeneous --tgt_cpu 1  (In-Order core)
#
# Note: dacapo runs are only valid for server <-> desktop pairs
#       (arm_edge_heterogeneous has no DaCapo data).

# =============================================================================
# ARM Server -> ARM Desktop
# =============================================================================

# --- all ---
# python3 cross_sys_arm.py --src_data_dir ../../../../processed_data_10M/arm_server/ --tgt_data_dir ../../../../processed_data_10M/arm_desktop/ --out_dir ../../../../results/cross_platform/cross_system/arm_server_to_arm_desktop_10M_all/ --strict_loocv --suite all
# python3 cross_sys_arm.py --src_data_dir ../../../../processed_data_100M/arm_server/ --tgt_data_dir ../../../../processed_data_100M/arm_desktop/ --out_dir ../../../../results/cross_platform/cross_system/arm_server_to_arm_desktop_100M_all/ --strict_loocv --suite all
# python3 cross_sys_arm.py --src_data_dir ../../../../processed_data_1000M/arm_server/ --tgt_data_dir ../../../../processed_data_1000M/arm_desktop/ --out_dir ../../../../results/cross_platform/cross_system/arm_server_to_arm_desktop_1000M_all/ --strict_loocv --suite all

# --- dacapo ---
# python3 cross_sys_arm.py --src_data_dir ../../../../processed_data_10M/arm_server/ --tgt_data_dir ../../../../processed_data_10M/arm_desktop/ --out_dir ../../../../results/cross_platform/cross_system/arm_server_to_arm_desktop_10M_dacapo/ --strict_loocv --suite dacapo
# python3 cross_sys_arm.py --src_data_dir ../../../../processed_data_100M/arm_server/ --tgt_data_dir ../../../../processed_data_100M/arm_desktop/ --out_dir ../../../../results/cross_platform/cross_system/arm_server_to_arm_desktop_100M_dacapo/ --strict_loocv --suite dacapo
# python3 cross_sys_arm.py --src_data_dir ../../../../processed_data_1000M/arm_server/ --tgt_data_dir ../../../../processed_data_1000M/arm_desktop/ --out_dir ../../../../results/cross_platform/cross_system/arm_server_to_arm_desktop_1000M_dacapo/ --strict_loocv --suite dacapo

# --- spec ---
# python3 cross_sys_arm.py --src_data_dir ../../../../processed_data_10M/arm_server/ --tgt_data_dir ../../../../processed_data_10M/arm_desktop/ --out_dir ../../../../results/cross_platform/cross_system/arm_server_to_arm_desktop_10M_spec/ --strict_loocv --freq 1.0 --suite spec
# python3 cross_sys_arm.py --src_data_dir ../../../../processed_data_100M/arm_server/ --tgt_data_dir ../../../../processed_data_100M/arm_desktop/ --out_dir ../../../../results/cross_platform/cross_system/arm_server_to_arm_desktop_100M_spec/ --strict_loocv --freq 1.0 --suite spec
python3 cross_sys_arm.py --src_data_dir ../../../../processed_data_1000M/arm_server/ --tgt_data_dir ../../../../processed_data_1000M/arm_desktop/ --out_dir ../../../../results/cross_platform/cross_system/arm_server_to_arm_desktop_1000M_spec_3/ --strict_loocv --freq 1.0 --suite spec --rolling_window 3

# =============================================================================
# ARM Desktop -> ARM Server
# =============================================================================

# --- all ---
# python3 cross_sys_arm.py --src_data_dir ../../../../processed_data_10M/arm_desktop/ --tgt_data_dir ../../../../processed_data_10M/arm_server/ --out_dir ../../../../results/cross_platform/cross_system/arm_desktop_to_arm_server_10M_all/ --strict_loocv --suite all
# python3 cross_sys_arm.py --src_data_dir ../../../../processed_data_100M/arm_desktop/ --tgt_data_dir ../../../../processed_data_100M/arm_server/ --out_dir ../../../../results/cross_platform/cross_system/arm_desktop_to_arm_server_100M_all/ --strict_loocv --suite all
# python3 cross_sys_arm.py --src_data_dir ../../../../processed_data_1000M/arm_desktop/ --tgt_data_dir ../../../../processed_data_1000M/arm_server/ --out_dir ../../../../results/cross_platform/cross_system/arm_desktop_to_arm_server_1000M_all/ --strict_loocv --suite all

# --- dacapo ---
# python3 cross_sys_arm.py --src_data_dir ../../../../processed_data_10M/arm_desktop/ --tgt_data_dir ../../../../processed_data_10M/arm_server/ --out_dir ../../../../results/cross_platform/cross_system/arm_desktop_to_arm_server_10M_dacapo/ --strict_loocv --suite dacapo
# python3 cross_sys_arm.py --src_data_dir ../../../../processed_data_100M/arm_desktop/ --tgt_data_dir ../../../../processed_data_100M/arm_server/ --out_dir ../../../../results/cross_platform/cross_system/arm_desktop_to_arm_server_100M_dacapo/ --strict_loocv --suite dacapo
# python3 cross_sys_arm.py --src_data_dir ../../../../processed_data_1000M/arm_desktop/ --tgt_data_dir ../../../../processed_data_1000M/arm_server/ --out_dir ../../../../results/cross_platform/cross_system/arm_desktop_to_arm_server_1000M_dacapo/ --strict_loocv --suite dacapo

# --- spec ---
# python3 cross_sys_arm.py --src_data_dir ../../../../processed_data_10M/arm_desktop/ --tgt_data_dir ../../../../processed_data_10M/arm_server/ --out_dir ../../../../results/cross_platform/cross_system/arm_desktop_to_arm_server_10M_spec/ --strict_loocv --freq 1.0 --suite spec
# python3 cross_sys_arm.py --src_data_dir ../../../../processed_data_100M/arm_desktop/ --tgt_data_dir ../../../../processed_data_100M/arm_server/ --out_dir ../../../../results/cross_platform/cross_system/arm_desktop_to_arm_server_100M_spec/ --strict_loocv --freq 1.0 --suite spec
python3 cross_sys_arm.py --src_data_dir ../../../../processed_data_1000M/arm_desktop/ --tgt_data_dir ../../../../processed_data_1000M/arm_server/ --out_dir ../../../../results/cross_platform/cross_system/arm_desktop_to_arm_server_1000M_spec_3/ --strict_loocv --freq 1.0 --suite spec --rolling_window 3

# =============================================================================
# ARM Server -> ARM Edge Out-of-Order (cpu4)
# =============================================================================

# --- all ---
# python3 cross_sys_arm.py --src_data_dir ../../../../processed_data_10M/arm_server/ --tgt_data_dir ../../../../processed_data_10M/arm_edge_heterogeneous/ --tgt_cpu 4 --out_dir ../../../../results/cross_platform/cross_system/arm_server_to_arm_edge_ooo_10M_all/ --strict_loocv --suite all
# python3 cross_sys_arm.py --src_data_dir ../../../../processed_data_100M/arm_server/ --tgt_data_dir ../../../../processed_data_100M/arm_edge_heterogeneous/ --tgt_cpu 4 --out_dir ../../../../results/cross_platform/cross_system/arm_server_to_arm_edge_ooo_100M_all/ --strict_loocv --suite all
# python3 cross_sys_arm.py --src_data_dir ../../../../processed_data_1000M/arm_server/ --tgt_data_dir ../../../../processed_data_1000M/arm_edge_heterogeneous/ --tgt_cpu 4 --out_dir ../../../../results/cross_platform/cross_system/arm_server_to_arm_edge_ooo_1000M_all/ --strict_loocv --suite all

# --- spec ---
# python3 cross_sys_arm.py --src_data_dir ../../../../processed_data_10M/arm_server/ --tgt_data_dir ../../../../processed_data_10M/arm_edge_heterogeneous/ --tgt_cpu 4 --out_dir ../../../../results/cross_platform/cross_system/arm_server_to_arm_edge_ooo_10M_spec/ --strict_loocv --freq 1.0 --suite spec
# python3 cross_sys_arm.py --src_data_dir ../../../../processed_data_100M/arm_server/ --tgt_data_dir ../../../../processed_data_100M/arm_edge_heterogeneous/ --tgt_cpu 4 --out_dir ../../../../results/cross_platform/cross_system/arm_server_to_arm_edge_ooo_100M_spec/ --strict_loocv --freq 1.0 --suite spec
python3 cross_sys_arm.py --src_data_dir ../../../../processed_data_1000M/arm_server/ --tgt_data_dir ../../../../processed_data_1000M/arm_edge_heterogeneous/ --tgt_cpu 4 --out_dir ../../../../results/cross_platform/cross_system/arm_server_to_arm_edge_ooo_1000M_spec_3/ --strict_loocv --freq 1.0 --suite spec --rolling_window 3

# =============================================================================
# ARM Edge Out-of-Order (cpu4) -> ARM Server
# =============================================================================

# --- all ---
# python3 cross_sys_arm.py --src_data_dir ../../../../processed_data_10M/arm_edge_heterogeneous/ --tgt_data_dir ../../../../processed_data_10M/arm_server/ --src_cpu 4 --out_dir ../../../../results/cross_platform/cross_system/arm_edge_ooo_to_arm_server_10M_all/ --strict_loocv --suite all
# python3 cross_sys_arm.py --src_data_dir ../../../../processed_data_100M/arm_edge_heterogeneous/ --tgt_data_dir ../../../../processed_data_100M/arm_server/ --src_cpu 4 --out_dir ../../../../results/cross_platform/cross_system/arm_edge_ooo_to_arm_server_100M_all/ --strict_loocv --suite all
# python3 cross_sys_arm.py --src_data_dir ../../../../processed_data_1000M/arm_edge_heterogeneous/ --tgt_data_dir ../../../../processed_data_1000M/arm_server/ --src_cpu 4 --out_dir ../../../../results/cross_platform/cross_system/arm_edge_ooo_to_arm_server_1000M_all/ --strict_loocv --suite all

# --- spec ---
# python3 cross_sys_arm.py --src_data_dir ../../../../processed_data_10M/arm_edge_heterogeneous/ --tgt_data_dir ../../../../processed_data_10M/arm_server/ --src_cpu 4 --out_dir ../../../../results/cross_platform/cross_system/arm_edge_ooo_to_arm_server_10M_spec/ --strict_loocv --freq 1.0 --suite spec
# python3 cross_sys_arm.py --src_data_dir ../../../../processed_data_100M/arm_edge_heterogeneous/ --tgt_data_dir ../../../../processed_data_100M/arm_server/ --src_cpu 4 --out_dir ../../../../results/cross_platform/cross_system/arm_edge_ooo_to_arm_server_100M_spec/ --strict_loocv --freq 1.0 --suite spec
python3 cross_sys_arm.py --src_data_dir ../../../../processed_data_1000M/arm_edge_heterogeneous/ --tgt_data_dir ../../../../processed_data_1000M/arm_server/ --src_cpu 4 --out_dir ../../../../results/cross_platform/cross_system/arm_edge_ooo_to_arm_server_1000M_spec_3/ --strict_loocv --freq 1.0 --suite spec --rolling_window 3

# =============================================================================
# ARM Server -> ARM Edge In-Order (cpu1)
# =============================================================================

# --- all ---
# python3 cross_sys_arm.py --src_data_dir ../../../../processed_data_10M/arm_server/ --tgt_data_dir ../../../../processed_data_10M/arm_edge_heterogeneous/ --tgt_cpu 1 --out_dir ../../../../results/cross_platform/cross_system/arm_server_to_arm_edge_ino_10M_all/ --strict_loocv --suite all
# python3 cross_sys_arm.py --src_data_dir ../../../../processed_data_100M/arm_server/ --tgt_data_dir ../../../../processed_data_100M/arm_edge_heterogeneous/ --tgt_cpu 1 --out_dir ../../../../results/cross_platform/cross_system/arm_server_to_arm_edge_ino_100M_all/ --strict_loocv --suite all
# python3 cross_sys_arm.py --src_data_dir ../../../../processed_data_1000M/arm_server/ --tgt_data_dir ../../../../processed_data_1000M/arm_edge_heterogeneous/ --tgt_cpu 1 --out_dir ../../../../results/cross_platform/cross_system/arm_server_to_arm_edge_ino_1000M_all/ --strict_loocv --suite all

# --- spec ---
# python3 cross_sys_arm.py --src_data_dir ../../../../processed_data_10M/arm_server/ --tgt_data_dir ../../../../processed_data_10M/arm_edge_heterogeneous/ --tgt_cpu 1 --out_dir ../../../../results/cross_platform/cross_system/arm_server_to_arm_edge_ino_10M_spec/ --strict_loocv --freq 1.0 --suite spec
# python3 cross_sys_arm.py --src_data_dir ../../../../processed_data_100M/arm_server/ --tgt_data_dir ../../../../processed_data_100M/arm_edge_heterogeneous/ --tgt_cpu 1 --out_dir ../../../../results/cross_platform/cross_system/arm_server_to_arm_edge_ino_100M_spec/ --strict_loocv --freq 1.0 --suite spec
python3 cross_sys_arm.py --src_data_dir ../../../../processed_data_1000M/arm_server/ --tgt_data_dir ../../../../processed_data_1000M/arm_edge_heterogeneous/ --tgt_cpu 1 --out_dir ../../../../results/cross_platform/cross_system/arm_server_to_arm_edge_ino_1000M_spec_3/ --strict_loocv --freq 1.0 --suite spec --rolling_window 3

# =============================================================================
# ARM Edge In-Order (cpu1) -> ARM Server
# =============================================================================

# --- all ---
# python3 cross_sys_arm.py --src_data_dir ../../../../processed_data_10M/arm_edge_heterogeneous/ --tgt_data_dir ../../../../processed_data_10M/arm_server/ --src_cpu 1 --out_dir ../../../../results/cross_platform/cross_system/arm_edge_ino_to_arm_server_10M_all/ --strict_loocv --suite all
# python3 cross_sys_arm.py --src_data_dir ../../../../processed_data_100M/arm_edge_heterogeneous/ --tgt_data_dir ../../../../processed_data_100M/arm_server/ --src_cpu 1 --out_dir ../../../../results/cross_platform/cross_system/arm_edge_ino_to_arm_server_100M_all/ --strict_loocv --suite all
# python3 cross_sys_arm.py --src_data_dir ../../../../processed_data_1000M/arm_edge_heterogeneous/ --tgt_data_dir ../../../../processed_data_1000M/arm_server/ --src_cpu 1 --out_dir ../../../../results/cross_platform/cross_system/arm_edge_ino_to_arm_server_1000M_all/ --strict_loocv --suite all

# --- spec ---
# python3 cross_sys_arm.py --src_data_dir ../../../../processed_data_10M/arm_edge_heterogeneous/ --tgt_data_dir ../../../../processed_data_10M/arm_server/ --src_cpu 1 --out_dir ../../../../results/cross_platform/cross_system/arm_edge_ino_to_arm_server_10M_spec/ --strict_loocv --freq 1.0 --suite spec
# python3 cross_sys_arm.py --src_data_dir ../../../../processed_data_100M/arm_edge_heterogeneous/ --tgt_data_dir ../../../../processed_data_100M/arm_server/ --src_cpu 1 --out_dir ../../../../results/cross_platform/cross_system/arm_edge_ino_to_arm_server_100M_spec/ --strict_loocv --freq 1.0 --suite spec
python3 cross_sys_arm.py --src_data_dir ../../../../processed_data_1000M/arm_edge_heterogeneous/ --tgt_data_dir ../../../../processed_data_1000M/arm_server/ --src_cpu 1 --out_dir ../../../../results/cross_platform/cross_system/arm_edge_ino_to_arm_server_1000M_spec_3/ --strict_loocv --freq 1.0 --suite spec --rolling_window 3

# =============================================================================
# ARM Desktop -> ARM Edge Out-of-Order (cpu4)
# =============================================================================

# --- all ---
# python3 cross_sys_arm.py --src_data_dir ../../../../processed_data_10M/arm_desktop/ --tgt_data_dir ../../../../processed_data_10M/arm_edge_heterogeneous/ --tgt_cpu 4 --out_dir ../../../../results/cross_platform/cross_system/arm_desktop_to_arm_edge_ooo_10M_all/ --strict_loocv --suite all
# python3 cross_sys_arm.py --src_data_dir ../../../../processed_data_100M/arm_desktop/ --tgt_data_dir ../../../../processed_data_100M/arm_edge_heterogeneous/ --tgt_cpu 4 --out_dir ../../../../results/cross_platform/cross_system/arm_desktop_to_arm_edge_ooo_100M_all/ --strict_loocv --suite all
# python3 cross_sys_arm.py --src_data_dir ../../../../processed_data_1000M/arm_desktop/ --tgt_data_dir ../../../../processed_data_1000M/arm_edge_heterogeneous/ --tgt_cpu 4 --out_dir ../../../../results/cross_platform/cross_system/arm_desktop_to_arm_edge_ooo_1000M_all/ --strict_loocv --suite all

# --- spec ---
# python3 cross_sys_arm.py --src_data_dir ../../../../processed_data_10M/arm_desktop/ --tgt_data_dir ../../../../processed_data_10M/arm_edge_heterogeneous/ --tgt_cpu 4 --out_dir ../../../../results/cross_platform/cross_system/arm_desktop_to_arm_edge_ooo_10M_spec/ --strict_loocv --freq 1.0 --suite spec
# python3 cross_sys_arm.py --src_data_dir ../../../../processed_data_100M/arm_desktop/ --tgt_data_dir ../../../../processed_data_100M/arm_edge_heterogeneous/ --tgt_cpu 4 --out_dir ../../../../results/cross_platform/cross_system/arm_desktop_to_arm_edge_ooo_100M_spec/ --strict_loocv --freq 1.0 --suite spec
python3 cross_sys_arm.py --src_data_dir ../../../../processed_data_1000M/arm_desktop/ --tgt_data_dir ../../../../processed_data_1000M/arm_edge_heterogeneous/ --tgt_cpu 4 --out_dir ../../../../results/cross_platform/cross_system/arm_desktop_to_arm_edge_ooo_1000M_spec_3/ --strict_loocv --freq 1.0 --suite spec --rolling_window 3

# =============================================================================
# ARM Edge Out-of-Order (cpu4) -> ARM Desktop
# =============================================================================

# --- all ---
# python3 cross_sys_arm.py --src_data_dir ../../../../processed_data_10M/arm_edge_heterogeneous/ --tgt_data_dir ../../../../processed_data_10M/arm_desktop/ --src_cpu 4 --out_dir ../../../../results/cross_platform/cross_system/arm_edge_ooo_to_arm_desktop_10M_all/ --strict_loocv --suite all
# python3 cross_sys_arm.py --src_data_dir ../../../../processed_data_100M/arm_edge_heterogeneous/ --tgt_data_dir ../../../../processed_data_100M/arm_desktop/ --src_cpu 4 --out_dir ../../../../results/cross_platform/cross_system/arm_edge_ooo_to_arm_desktop_100M_all/ --strict_loocv --suite all
# python3 cross_sys_arm.py --src_data_dir ../../../../processed_data_1000M/arm_edge_heterogeneous/ --tgt_data_dir ../../../../processed_data_1000M/arm_desktop/ --src_cpu 4 --out_dir ../../../../results/cross_platform/cross_system/arm_edge_ooo_to_arm_desktop_1000M_all/ --strict_loocv --suite all

# --- spec ---
# python3 cross_sys_arm.py --src_data_dir ../../../../processed_data_10M/arm_edge_heterogeneous/ --tgt_data_dir ../../../../processed_data_10M/arm_desktop/ --src_cpu 4 --out_dir ../../../../results/cross_platform/cross_system/arm_edge_ooo_to_arm_desktop_10M_spec/ --strict_loocv --freq 1.0 --suite spec
# python3 cross_sys_arm.py --src_data_dir ../../../../processed_data_100M/arm_edge_heterogeneous/ --tgt_data_dir ../../../../processed_data_100M/arm_desktop/ --src_cpu 4 --out_dir ../../../../results/cross_platform/cross_system/arm_edge_ooo_to_arm_desktop_100M_spec/ --strict_loocv --freq 1.0 --suite spec
python3 cross_sys_arm.py --src_data_dir ../../../../processed_data_1000M/arm_edge_heterogeneous/ --tgt_data_dir ../../../../processed_data_1000M/arm_desktop/ --src_cpu 4 --out_dir ../../../../results/cross_platform/cross_system/arm_edge_ooo_to_arm_desktop_1000M_spec_3/ --strict_loocv --freq 1.0 --suite spec --rolling_window 3

# =============================================================================
# ARM Desktop -> ARM Edge In-Order (cpu1)
# =============================================================================

# --- all ---
# python3 cross_sys_arm.py --src_data_dir ../../../../processed_data_10M/arm_desktop/ --tgt_data_dir ../../../../processed_data_10M/arm_edge_heterogeneous/ --tgt_cpu 1 --out_dir ../../../../results/cross_platform/cross_system/arm_desktop_to_arm_edge_ino_10M_all/ --strict_loocv --suite all
# python3 cross_sys_arm.py --src_data_dir ../../../../processed_data_100M/arm_desktop/ --tgt_data_dir ../../../../processed_data_100M/arm_edge_heterogeneous/ --tgt_cpu 1 --out_dir ../../../../results/cross_platform/cross_system/arm_desktop_to_arm_edge_ino_100M_all/ --strict_loocv --suite all
# python3 cross_sys_arm.py --src_data_dir ../../../../processed_data_1000M/arm_desktop/ --tgt_data_dir ../../../../processed_data_1000M/arm_edge_heterogeneous/ --tgt_cpu 1 --out_dir ../../../../results/cross_platform/cross_system/arm_desktop_to_arm_edge_ino_1000M_all/ --strict_loocv --suite all

# --- spec ---
# python3 cross_sys_arm.py --src_data_dir ../../../../processed_data_10M/arm_desktop/ --tgt_data_dir ../../../../processed_data_10M/arm_edge_heterogeneous/ --tgt_cpu 1 --out_dir ../../../../results/cross_platform/cross_system/arm_desktop_to_arm_edge_ino_10M_spec/ --strict_loocv --freq 1.0 --suite spec
# python3 cross_sys_arm.py --src_data_dir ../../../../processed_data_100M/arm_desktop/ --tgt_data_dir ../../../../processed_data_100M/arm_edge_heterogeneous/ --tgt_cpu 1 --out_dir ../../../../results/cross_platform/cross_system/arm_desktop_to_arm_edge_ino_100M_spec/ --strict_loocv --freq 1.0 --suite spec
python3 cross_sys_arm.py --src_data_dir ../../../../processed_data_1000M/arm_desktop/ --tgt_data_dir ../../../../processed_data_1000M/arm_edge_heterogeneous/ --tgt_cpu 1 --out_dir ../../../../results/cross_platform/cross_system/arm_desktop_to_arm_edge_ino_1000M_spec_3/ --strict_loocv --freq 1.0 --suite spec --rolling_window 3

# =============================================================================
# ARM Edge In-Order (cpu1) -> ARM Desktop
# =============================================================================

# --- all ---
# python3 cross_sys_arm.py --src_data_dir ../../../../processed_data_10M/arm_edge_heterogeneous/ --tgt_data_dir ../../../../processed_data_10M/arm_desktop/ --src_cpu 1 --out_dir ../../../../results/cross_platform/cross_system/arm_edge_ino_to_arm_desktop_10M_all/ --strict_loocv --suite all
# python3 cross_sys_arm.py --src_data_dir ../../../../processed_data_100M/arm_edge_heterogeneous/ --tgt_data_dir ../../../../processed_data_100M/arm_desktop/ --src_cpu 1 --out_dir ../../../../results/cross_platform/cross_system/arm_edge_ino_to_arm_desktop_100M_all/ --strict_loocv --suite all
# python3 cross_sys_arm.py --src_data_dir ../../../../processed_data_1000M/arm_edge_heterogeneous/ --tgt_data_dir ../../../../processed_data_1000M/arm_desktop/ --src_cpu 1 --out_dir ../../../../results/cross_platform/cross_system/arm_edge_ino_to_arm_desktop_1000M_all/ --strict_loocv --suite all

# --- spec ---
# python3 cross_sys_arm.py --src_data_dir ../../../../processed_data_10M/arm_edge_heterogeneous/ --tgt_data_dir ../../../../processed_data_10M/arm_desktop/ --src_cpu 1 --out_dir ../../../../results/cross_platform/cross_system/arm_edge_ino_to_arm_desktop_10M_spec/ --strict_loocv --freq 1.0 --suite spec
# python3 cross_sys_arm.py --src_data_dir ../../../../processed_data_100M/arm_edge_heterogeneous/ --tgt_data_dir ../../../../processed_data_100M/arm_desktop/ --src_cpu 1 --out_dir ../../../../results/cross_platform/cross_system/arm_edge_ino_to_arm_desktop_100M_spec/ --strict_loocv --freq 1.0 --suite spec
python3 cross_sys_arm.py --src_data_dir ../../../../processed_data_1000M/arm_edge_heterogeneous/ --tgt_data_dir ../../../../processed_data_1000M/arm_desktop/ --src_cpu 1 --out_dir ../../../../results/cross_platform/cross_system/arm_edge_ino_to_arm_desktop_1000M_spec_3/ --strict_loocv --freq 1.0 --suite spec --rolling_window 3

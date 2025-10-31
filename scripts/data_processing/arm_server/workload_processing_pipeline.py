#!/usr/bin/env python3
"""
Parallel Processing Pipeline for SPEC and DaCapo Benchmarks
Processes data in stages with barriers between each step:
1. Perf script conversion (raw .out -> raw CSVs)
2. AWK trimming (raw CSVs -> trimmed CSVs)
3. Pandas pivoting/merging (trimmed CSVs -> premerge CSVs)
4. Final CSV generation (premerge CSVs -> final CSVs)
"""

import subprocess
import multiprocessing as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import uuid
import time
import sys

class ParallelPipeline:
    def __init__(self, base_dir, max_workers=None):
        self.base_dir = Path(base_dir)
        self.raw_data_dir = self.base_dir / "raw_data"
        self.raw_csvs_dir = self.base_dir / "raw_csvs"
        self.trimmed_csvs_dir = self.base_dir / "trimmed_csvs"
        self.premerge_csvs_dir = self.base_dir / "premerge_csvs"
        self.final_csvs_dir = self.base_dir / "final_csvs"
        
        # Create directories
        for d in [self.raw_csvs_dir, self.trimmed_csvs_dir, self.premerge_csvs_dir, self.final_csvs_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Use all available CPUs if not specified
        self.max_workers = max_workers or mp.cpu_count()
        
        print(f" Parallel Pipeline initialized with {self.max_workers} workers")
        print(f" Base directory: {self.base_dir}")
    
    def find_raw_files(self):
        """Find all raw .out files"""
        raw_files = list(self.raw_data_dir.glob("*.out"))
        return sorted(raw_files)
    
    # ============================================================================
    # STAGE 1: PERF SCRIPT CONVERSION (RAW .OUT -> RAW CSV)
    # ============================================================================
    
    def process_perf_file(self, out_file):
        """Convert a single .out file to CSV using perf script"""
        try:
            csv_file = self.raw_csvs_dir / f"{out_file.stem}.csv"
            
            # Skip if already processed
            if csv_file.exists():
                return f" {out_file.name} (already exists)"
            
            # AWK script to parse perf script output
            awk_script = """
BEGIN {
    print "timestamp,event,count,comm,pid,cpu"
}
{
    timestamp = $4
    event = $6
    count = $5
    comm = $1
    pid = $2
    cpu = $3
    
    if (timestamp != "time" && timestamp != "" && event != "" && count != "") {
        print timestamp "," event "," count "," comm "," pid "," cpu
    }
}
"""
            
            awk_file = Path(f"/tmp/temp_awk_{uuid.uuid4().hex}.awk")
            with open(awk_file, 'w') as f:
                f.write(awk_script)
            
            # Run perf script | awk > csv
            with open(csv_file, 'w') as csv_f:
                perf_proc = subprocess.Popen(
                    ['perf', 'script', '-i', str(out_file), '-F', 'comm,pid,cpu,time,period,event'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                awk_proc = subprocess.Popen(
                    ['awk', '-f', str(awk_file)],
                    stdin=perf_proc.stdout,
                    stdout=csv_f,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                perf_proc.stdout.close()
                awk_proc.communicate()
                perf_proc.communicate()
            
            awk_file.unlink(missing_ok=True)
            
            if perf_proc.returncode != 0 or awk_proc.returncode != 0:
                csv_file.unlink(missing_ok=True)
                return f" {out_file.name} (perf/awk failed)"
            
            return f" {out_file.name}"
            
        except Exception as e:
            return f" {out_file.name} ({str(e)})"
    
    def stage1_perf_conversion(self):
        """Stage 1: Convert all .out files to CSV using perf script"""
        print("\n" + "="*80)
        print("STAGE 1: PERF SCRIPT CONVERSION (.out -> CSV)")
        print("="*80)
        
        raw_files = self.find_raw_files()
        total_files = len(raw_files)
        
        print(f" Found {total_files} raw .out files")
        print(f" Processing with {self.max_workers} parallel workers...\n")
        
        start_time = time.time()
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.process_perf_file, f): f for f in raw_files}
            
            completed = 0
            for future in as_completed(futures):
                result = future.result()
                completed += 1
                print(f"[{completed}/{total_files}] {result}")
        
        elapsed = time.time() - start_time
        print(f"\n Stage 1 complete in {elapsed:.1f}s")
        print("="*80)
    
    # ============================================================================
    # STAGE 2: AWK TRIMMING (RAW CSV -> TRIMMED CSV)
    # ============================================================================
    
    def get_main_process_for_workload(self, filename):
        """Determine the main process name for filtering"""
        # SPEC workloads
        spec_processes = {
            '500': 'perlbench_r_bas',
            '502': 'cpugcc_r_base.m',
            '503': 'bwaves_r_base.m',
            '505': 'mcf_r_base.myte',
            '507': 'cactusBSSN_r_ba',
            '508': 'namd_r_base.myt',
            '510': 'parest_r_base.m',
            '511': 'povray_r_base.m',
            '519': 'lbm_r_base.myte',
            '520': 'omnetpp_r_base.',
            '521': 'wrf_r_base.myte',
            '523': 'cpuxalan_r_base',
            '525': 'x264_r_base.myt',
            '526': 'blender_r_base.',
            '527': 'cam4_r_base.myt',
            '531': 'deepsjeng_r_bas',
            '538': 'imagick_r_base.',
            '541': 'leela_r_base.my',
            '544': 'nab_r_base.myte',
            '548': 'exchange2_r_bas',
            '549': 'fotonik3d_r_ba',
            '554': 'roms_r_base.myt',
            '557': 'xz_r_base.mytes'
        }
        
        # Check if it's a SPEC workload
        for spec_num, proc_name in spec_processes.items():
            if f'spec_{spec_num}_' in filename:
                return proc_name
        
        # DaCapo workloads - filter by "java" process
        if 'dacapo_' in filename:
            return 'java'
        
        return None
    
    def trim_csv_file(self, csv_file):
        """Trim a CSV file to only include main process events"""
        try:
            trimmed_file = self.trimmed_csvs_dir / f"{csv_file.stem}_trimmed.csv"
            
            # Skip if already processed
            if trimmed_file.exists():
                return f" {csv_file.name} (already exists)"
            
            main_process = self.get_main_process_for_workload(csv_file.name)
            
            if not main_process:
                return f" {csv_file.name} (unknown workload type)"
            
            # Read CSV and filter by main process
            df = pd.read_csv(csv_file)
            
            # Filter for main process
            df_trimmed = df[df['comm'].str.contains(main_process, na=False)]
            
            if df_trimmed.empty:
                return f" {csv_file.name} (no matching process '{main_process}')"
            
            # Save trimmed CSV
            df_trimmed.to_csv(trimmed_file, index=False)
            
            return f" {csv_file.name} ({len(df)} -> {len(df_trimmed)} rows)"
            
        except Exception as e:
            return f" {csv_file.name} ({str(e)})"
    
    def stage2_awk_trimming(self):
        """Stage 2: Trim CSVs to only include main process"""
        print("\n" + "="*80)
        print("STAGE 2: TRIMMING (CSV -> TRIMMED CSV)")
        print("="*80)
        
        csv_files = list(self.raw_csvs_dir.glob("*.csv"))
        total_files = len(csv_files)
        
        print(f" Found {total_files} raw CSV files")
        print(f" Processing with {self.max_workers} parallel workers...\n")
        
        start_time = time.time()
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.trim_csv_file, f): f for f in csv_files}
            
            completed = 0
            for future in as_completed(futures):
                result = future.result()
                completed += 1
                print(f"[{completed}/{total_files}] {result}")
        
        elapsed = time.time() - start_time
        print(f"\n Stage 2 complete in {elapsed:.1f}s")
        print("="*80)
    
    # ============================================================================
    # STAGE 3: PANDAS PIVOTING (TRIMMED CSV -> PREMERGE CSV)
    # ============================================================================
    
    def pivot_trimmed_file(self, trimmed_file):
        """Convert a single trimmed CSV to premerge format (pivot on events)"""
        try:
            premerge_file = self.premerge_csvs_dir / trimmed_file.name.replace('_trimmed.csv', '_premerge.csv')
            
            # Skip if already processed
            if premerge_file.exists():
                return f" {trimmed_file.name} (already exists)"
            
            # Read trimmed CSV
            df = pd.read_csv(trimmed_file)
            
            # Pivot: each event becomes a column
            # Keep timestamp as index, events as columns
            if 'event' in df.columns and 'count' in df.columns and 'timestamp' in df.columns:
                # Pivot table with events as columns
                pivoted = df.pivot_table(
                    index='timestamp',
                    columns='event',
                    values='count',
                    aggfunc='first'
                ).reset_index()
                
                # Save premerge file
                pivoted.to_csv(premerge_file, index=False)
                
                return f" {trimmed_file.name} ({len(df)} -> {len(pivoted)} rows, {len(pivoted.columns)} cols)"
            else:
                # No pivoting needed, just copy
                df.to_csv(premerge_file, index=False)
                return f" {trimmed_file.name} (copied, {len(df)} rows)"
            
        except Exception as e:
            return f" {trimmed_file.name} ({str(e)})"
    
    def stage3_pandas_pivoting(self):
        """Stage 3: Pivot trimmed CSVs (each trimmed file -> one premerge file)"""
        print("\n" + "="*80)
        print("STAGE 3: PANDAS PIVOTING (TRIMMED CSV -> PREMERGE CSV)")
        print("="*80)
        
        trimmed_files = list(self.trimmed_csvs_dir.glob("*_trimmed.csv"))
        total_files = len(trimmed_files)
        
        print(f" Found {total_files} trimmed CSV files")
        print(f" Processing with {self.max_workers} parallel workers...\n")
        
        start_time = time.time()
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.pivot_trimmed_file, f): f for f in trimmed_files}
            
            completed = 0
            for future in as_completed(futures):
                result = future.result()
                completed += 1
                print(f"[{completed}/{total_files}] {result}")
        
        elapsed = time.time() - start_time
        print(f"\n Stage 3 complete in {elapsed:.1f}s")
        print("="*80)
    
    # ============================================================================
    # STAGE 4: FINAL CSV GENERATION (MERGE ALL COUNTER GROUPS)
    # ============================================================================
    
    def group_premerge_files(self):
        """Group premerge files by workload (all counter groups together)"""
        from collections import defaultdict
        
        premerge_files = list(self.premerge_csvs_dir.glob("*_premerge.csv"))
        
        workload_groups = defaultdict(list)
        
        for f in premerge_files:
            # Parse filename to extract workload identifier
            # Two formats:
            # DaCapo: cpu_0_1.5GHz_dacapo_avrora_10000000_{counter_group}_premerge.csv
            # SPEC:   cpu_0_1.5GHz_spec_520_10000000_{counter_group}_1_premerge.csv
            parts = f.stem.split('_')
            
            # Check if this is SPEC or DaCapo based on presence of '_1' before '_premerge'
            # parts[-1] is 'premerge', parts[-2] is either counter_group (DaCapo) or '1' (SPEC)
            
            if 'spec_' in f.name:
                # SPEC format: counter_group is at parts[-3], skip parts[-2] which is always '1'
                counter_group_idx = len(parts) - 3
                workload_key = '_'.join(parts[:counter_group_idx])
            else:
                # DaCapo format: counter_group is at parts[-2]
                counter_group_idx = len(parts) - 2
                workload_key = '_'.join(parts[:counter_group_idx])
            
            # Get counter group number
            try:
                counter_group_num = int(parts[counter_group_idx])
                workload_groups[workload_key].append((counter_group_num, f))
            except ValueError:
                print(f"Warning: Could not parse counter group from {f.name}")
                continue
        
        # Sort each group by counter group number
        for key in workload_groups:
            workload_groups[key].sort(key=lambda x: x[0])
        
        return workload_groups
    
    def merge_counter_groups(self, workload_key, counter_group_files):
        """Merge all counter groups for a single workload into one final CSV"""
        try:
            # Determine frequency-based subdirectory
            if '1.5GHz' in workload_key:
                freq_dir = self.final_csvs_dir / "1.5GHz"
            elif '3.0GHz' in workload_key:
                freq_dir = self.final_csvs_dir / "3.0GHz"
            else:
                freq_dir = self.final_csvs_dir  # Fallback to root
            
            # Create frequency subdirectory if it doesn't exist
            freq_dir.mkdir(parents=True, exist_ok=True)
            
            final_file = freq_dir / f"{workload_key}_merged.csv"
            
            # Skip if already processed
            if final_file.exists():
                return f" {workload_key} (already exists)"
            
            # Determine which groups to use based on workload type
            # SPEC 1.5GHz: groups 0-5, 10-12 (groups 6-9, 13-18 have different counters)
            # SPEC 3.0GHz and DaCapo: groups 0-8
            if '1.5GHz_spec_' in workload_key:
                # SPEC 1.5GHz: use groups 0-5, 10-12
                required_groups = [0, 1, 2, 3, 4, 5, 10, 11, 12]
            else:
                # SPEC 3.0GHz and DaCapo: use groups 0-8
                required_groups = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            
            # Filter to only use the required groups
            filtered_files = [(num, path) for num, path in counter_group_files if num in required_groups]
            
            if len(filtered_files) != len(required_groups):
                found_groups = [num for num, _ in filtered_files]
                return f" {workload_key} (missing groups, expected {required_groups}, found {found_groups})"
            
            # Load all counter group files
            dfs = []
            for group_num, filepath in filtered_files:
                df = pd.read_csv(filepath)
                dfs.append((group_num, df, filepath.name))
            
            if not dfs:
                return f" {workload_key} (no files found)"
            
            # Find the shortest dataframe to truncate all to the same length
            min_length = min(len(df) for _, df, _ in dfs)
            
            # Start with group 0 as the base
            base_df = None
            for group_num, df, fname in dfs:
                if group_num == 0:
                    base_df = df.iloc[:min_length].copy()
                    break
            
            if base_df is None:
                return f" {workload_key} (group 0 not found)"
            
            # Add sample_number column at the beginning
            base_df.insert(0, 'sample_number', range(len(base_df)))
            
            # Drop timestamp column (not needed for analysis)
            if 'timestamp' in base_df.columns:
                base_df = base_df.drop(columns=['timestamp'])
            
            # Track which columns we already have (normalize column names)
            existing_columns = set(col.strip().lower() for col in base_df.columns)
            
            # Merge other counter groups (skip group 0 since it's the base)
            for group_num, df, fname in dfs:
                if group_num == 0:
                    continue
                
                # Truncate to min_length
                df_truncated = df.iloc[:min_length].copy()
                
                # Add columns that don't already exist (case-insensitive check)
                for col in df_truncated.columns:
                    col_normalized = col.strip().lower()
                    
                    # Skip timestamp and instructions (already in base)
                    if col_normalized in ['timestamp', 'instructions:pp:', 'instructions']:
                        continue
                    
                    # Skip if column already exists
                    if col_normalized in existing_columns:
                        continue
                    
                    # Add new column
                    base_df[col] = df_truncated[col].values
                    existing_columns.add(col_normalized)
            
            # Save final merged CSV
            base_df.to_csv(final_file, index=False)
            
            num_groups = len(counter_group_files)
            num_cols = len(base_df.columns)
            
            return f" {workload_key} ({num_groups} groups merged, {min_length} rows, {num_cols} columns)"
            
        except Exception as e:
            import traceback
            return f" {workload_key} (ERROR: {str(e)}\n{traceback.format_exc()})"
    
    def stage4_final_generation(self):
        """Stage 4: Merge all counter groups for each workload into final CSVs"""
        print("\n" + "="*80)
        print("STAGE 4: FINAL CSV GENERATION (MERGE COUNTER GROUPS)")
        print("="*80)
        
        # Group premerge files by workload
        workload_groups = self.group_premerge_files()
        total_workloads = len(workload_groups)
        
        print(f" Found {total_workloads} unique workloads to merge")
        print(f" Processing with {self.max_workers} parallel workers...\n")
        
        start_time = time.time()
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.merge_counter_groups, key, files): key 
                for key, files in workload_groups.items()
            }
            
            completed = 0
            for future in as_completed(futures):
                result = future.result()
                completed += 1
                print(f"[{completed}/{total_workloads}] {result}")
        
        elapsed = time.time() - start_time
        print(f"\n Stage 4 complete in {elapsed:.1f}s")
        print("="*80)
    
    # ============================================================================
    # MAIN PIPELINE EXECUTION
    # ============================================================================
    
    def run_pipeline(self):
        """Run the complete pipeline with barriers between stages"""
        print("\n" + "="*80)
        print(" STARTING PARALLEL PROCESSING PIPELINE")
        print("="*80)
        print(f"Workers: {self.max_workers}")
        print(f"Base directory: {self.base_dir}")
        
        overall_start = time.time()
        
        try:
            # Stage 1: Perf script conversion
            self.stage1_perf_conversion()
            
            # Barrier (implicit - all Stage 1 processes complete before Stage 2 starts)
            print("\n BARRIER: All perf script conversions complete")
            
            # Stage 2: AWK trimming
            self.stage2_awk_trimming()
            
            # Barrier
            print("\n BARRIER: All trimming complete")
            
            # Stage 3: Pandas pivoting
            self.stage3_pandas_pivoting()
            
            # Barrier
            print("\n BARRIER: All pivoting complete")
            
            # Stage 4: Final generation
            self.stage4_final_generation()
            
            # Final summary
            overall_elapsed = time.time() - overall_start
            
            print("\n" + "="*80)
            print(" PIPELINE COMPLETE!")
            print("="*80)
            print(f"Total time: {overall_elapsed:.1f}s ({overall_elapsed/60:.1f} minutes)")
            
            # Count files in each directory
            raw_count = len(list(self.raw_data_dir.glob("*.out")))
            csv_count = len(list(self.raw_csvs_dir.glob("*.csv")))
            trimmed_count = len(list(self.trimmed_csvs_dir.glob("*_trimmed.csv")))
            premerge_count = len(list(self.premerge_csvs_dir.glob("*_premerge.csv")))
            final_count = len(list(self.final_csvs_dir.glob("*_merged.csv")))
            
            print(f"\n File counts:")
            print(f"  Raw .out files:      {raw_count}")
            print(f"  Raw CSVs:            {csv_count}")
            print(f"  Trimmed CSVs:        {trimmed_count}")
            print(f"  Premerge CSVs:       {premerge_count}")
            print(f"  Final merged CSVs:   {final_count}")
            print("\n Ready for variance analysis!")
            
        except Exception as e:
            print(f"\n Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


def main():
    base_dir = Path("../../../data/arm_server")
    
    # Create pipeline with all available CPU cores
    pipeline = ParallelPipeline(base_dir, max_workers=mp.cpu_count())
    
    # Run the complete pipeline
    pipeline.run_pipeline()


if __name__ == "__main__":
    main()


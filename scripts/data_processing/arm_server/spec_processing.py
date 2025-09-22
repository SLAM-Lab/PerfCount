#!/usr/bin/env python3
"""
Fixed SPEC Pipeline - Processes workloads in parallel but steps sequentially within each workload.
This prevents cross-contamination between different SPEC workloads.
"""

import pandas as pd
import sys
import os
import time
import threading
from pathlib import Path
import re
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

class FixedSPECPipeline:
    def __init__(self, base_dir, max_workers=4):
        self.base_dir = Path(base_dir)
        self.raw_data_dir = self.base_dir / "raw_data"
        self.raw_csvs_dir = self.base_dir / "raw_csvs"
        self.trimmed_csvs_dir = self.base_dir / "trimmed_csvs"
        self.premerge_csvs_dir = self.base_dir / "premerge_csvs"
        self.final_csvs_dir = self.base_dir / "final_csvs"
        self.max_workers = max_workers
        
        # Thread-safe printing
        self.print_lock = threading.Lock()
        
        # Main workload processes for each SPEC benchmark
        self.main_processes = {
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
            '527': 'cam4_r_base.myt',
            '531': 'deepsjeng_r_bas',
            '538': 'imagick_r_base.',
            '541': 'leela_r_base.my',
            '544': 'nab_r_base.myte',
            '548': 'exchange2_r_bas',
            '549': 'fotonik3d_r_bas',
            '554': 'roms_r_base.myt',
            '557': 'xz_r_base.mytes'
        }
    
    def thread_print(self, message):
        """Thread-safe printing."""
        with self.print_lock:
            print(message)
    
    def setup_directories(self):
        """Create necessary directories."""
        for directory in [self.raw_data_dir, self.raw_csvs_dir, self.trimmed_csvs_dir, 
                         self.premerge_csvs_dir, self.final_csvs_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def convert_out_to_csv(self, out_file):
        """Convert a single .out file to CSV using perf script + awk."""
        csv_file = self.raw_csvs_dir / (out_file.stem + ".csv")
        
        if csv_file.exists():
            return str(csv_file)
        
        try:
            # Use perf script with specific fields and pipe to awk for CSV conversion
            cmd = ['perf', 'script', '-i', str(out_file), '-F', 'comm,pid,cpu,time,period,event,ip,sym']
            
            awk_script = '''
            BEGIN {
                print "process,pid,cpu,timestamp,event_count,event_name,address,symbol"
            }
            {
                process = $1
                pid = $2
                cpu = $3
                gsub(/\\[|\\]/, "", cpu)
                timestamp = $4
                gsub(/:/, "", timestamp)
                
                colon_pos = index($0, ":")
                if (colon_pos > 0) {
                    event_part = substr($0, colon_pos + 1)
                    gsub(/^[ \\t]+/, "", event_part)
                    n = split(event_part, event_fields, " ")
                    
                    if (n >= 3) {
                        event_count = event_fields[1]
                        event_name = event_fields[2]
                        gsub(/:/, "", event_name)
                        address = event_fields[3]
                        
                        symbol = ""
                        for (i = 4; i <= n; i++) {
                            if (symbol != "") symbol = symbol " "
                            symbol = symbol event_fields[i]
                        }
                        if (symbol == "") symbol = "[unknown]"
                        
                        # Quote fields that contain commas to avoid CSV parsing issues
                        if (process ~ /,/) process = "\"" process "\""
                        if (symbol ~ /,/) symbol = "\"" symbol "\""
                        
                        print process "," pid "," cpu "," timestamp "," event_count "," event_name "," address "," symbol
                    }
                }
            }'''
            
            # Write awk script to temporary file with unique name to avoid race conditions
            awk_file = self.base_dir / f"temp_awk_{uuid.uuid4().hex[:8]}.awk"
            with open(awk_file, 'w') as f:
                f.write(awk_script)
            
            # Execute the command
            import subprocess
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                return None
            
            # Process with awk
            awk_result = subprocess.run(['awk', '-f', str(awk_file)], 
                                      input=result.stdout, capture_output=True, text=True)
            
            # Clean up temp file
            awk_file.unlink()
            
            if awk_result.returncode != 0:
                return None
            
            # Write to CSV file
            with open(csv_file, 'w') as f:
                f.write(awk_result.stdout)
            
            return str(csv_file)
            
        except Exception as e:
            return None
    
    def trim_csv_to_workload(self, csv_file):
        """Trim CSV to only include main workload process."""
        trimmed_file = self.trimmed_csvs_dir / (csv_file.stem + "_trimmed.csv")
        
        if trimmed_file.exists():
            return str(trimmed_file)
        
        try:
            # Extract SPEC number from filename
            spec_num = None
            filename = csv_file.name
            if 'spec_' in filename:
                match = re.search(r'spec_(\d+)', filename)
                if match:
                    spec_num = match.group(1)
            
            if not spec_num:
                return None
            
            main_process = self.main_processes.get(spec_num)
            if not main_process:
                return None

            # Use Python with pandas for robust CSV parsing
            df = pd.read_csv(csv_file, low_memory=False, on_bad_lines='skip')
            if df.empty:
                return None
            
            # Filter to main workload process
            filtered_df = df[df['process'] == main_process]
            if filtered_df.empty:
                return None
            
            filtered_df.to_csv(trimmed_file, index=False)
            return str(trimmed_file)
            
        except Exception as e:
            return None
    
    def create_premerge_file(self, trimmed_file):
        """Create pre-merge file from trimmed CSV."""
        premerge_file = self.premerge_csvs_dir / (trimmed_file.stem.replace('_trimmed', '_premerge') + '.csv')
        
        if premerge_file.exists():
            return str(premerge_file)
        
        try:
            df = pd.read_csv(trimmed_file, low_memory=False, on_bad_lines='skip')
            if df.empty:
                return None
            
            # Pivot the table to have event_name as columns
            df_pivot = df.pivot_table(
                index='timestamp',
                columns='event_name',
                values='event_count',
                aggfunc='first'
            ).reset_index()
            
            df_pivot.columns.name = None
            df_pivot.insert(0, 'sample_number', range(1, 1 + len(df_pivot)))
            
            # Reorder columns
            cols = ['sample_number', 'timestamp', 'instructions']
            other_cols = [col for col in df_pivot.columns if col not in cols]
            cols.extend(sorted(other_cols))
            df_pivot = df_pivot[cols]
            
            df_pivot.to_csv(premerge_file, index=False)
            return str(premerge_file)
            
        except Exception as e:
            return None
    
    def extract_metadata_from_filename(self, filename):
        """Extract frequency and collection granularity from filename."""
        # Expected format: cpu_0_3.0GHz_spec_500_10000000_0.out
        # Extract frequency (3.0GHz) and granularity (10000000)
        freq_match = re.search(r'cpu_0_(\d+\.\d+GHz)', filename)
        granularity_match = re.search(r'spec_\d+_(\d+)_\d+', filename)
        
        frequency = freq_match.group(1) if freq_match else "3.0GHz"
        granularity = granularity_match.group(1) if granularity_match else "10000000"
        
        return frequency, granularity

    def merge_workload_files(self, spec_num, premerge_files):
        """Merge all pre-merge files for a workload."""
        if not premerge_files:
            return None
        
        try:
            # Extract metadata from the first premerge file name
            first_file = Path(premerge_files[0])
            frequency, granularity = self.extract_metadata_from_filename(first_file.name)
            
            # Load base file
            df_merged = pd.read_csv(premerge_files[0])
            df_merged = df_merged.drop('timestamp', axis=1)
            min_length = len(df_merged)
            
            # Merge subsequent files
            for file_path in premerge_files[1:]:
                df_file = pd.read_csv(file_path)
                cols_to_drop = ['timestamp']
                if 'instructions' in df_file.columns:
                    cols_to_drop.append('instructions')
                if 'sample_number' in df_file.columns:
                    cols_to_drop.append('sample_number')
                
                df_file_clean = df_file.drop(columns=cols_to_drop, errors='ignore')
                df_merged = pd.concat([df_merged, df_file_clean], axis=1)
                min_length = min(min_length, len(df_file_clean))
            
            # Trim to shortest length
            df_merged = df_merged.iloc[:min_length]
            
            # Add metadata columns
            df_merged.insert(0, 'frequency', frequency)
            df_merged.insert(1, 'collection_granularity', granularity)
            df_merged.insert(2, 'spec_number', spec_num)
            
            # Convert to integers (except metadata columns)
            for col in df_merged.columns:
                if col not in ['frequency', 'collection_granularity', 'spec_number', 'sample_number']:
                    df_merged[col] = df_merged[col].fillna(0).astype('int64')
            
            # Save merged file
            output_file = self.final_csvs_dir / f"spec_{spec_num}_merged.csv"
            df_merged.to_csv(output_file, index=False)
            
            return str(output_file)
            
        except Exception as e:
            return None
    
    def process_workload(self, spec_num):
        """Process a single SPEC workload end-to-end (sequential steps)."""
        start_time = time.time()
        
        # Find .out files for this SPEC
        out_files = sorted(list(self.raw_data_dir.glob(f"cpu_0_3.0GHz_spec_{spec_num}_*.out")))
        if not out_files:
            self.thread_print(f"SPEC {spec_num}: No .out files found")
            return None
        
        # Step 1: Convert .out files to CSV (sequential)
        csv_files = []
        
        for out_file in out_files:
            csv_file = self.convert_out_to_csv(out_file)
            if csv_file:
                csv_files.append(csv_file)
        
        if not csv_files:
            self.thread_print(f"SPEC {spec_num}: No CSV files created")
            return None
        
        self.thread_print(f"SPEC {spec_num}: ✅ Raw CSVs completed ({len(csv_files)}/{len(out_files)} files)")
        
        # Step 2: Trim CSV files (sequential)
        trimmed_files = []
        
        for csv_file in csv_files:
            trimmed_file = self.trim_csv_to_workload(Path(csv_file))
            if trimmed_file:
                trimmed_files.append(trimmed_file)
        
        if not trimmed_files:
            self.thread_print(f"SPEC {spec_num}: No trimmed files created")
            return None
        
        self.thread_print(f"SPEC {spec_num}: ✅ Trimmed CSVs completed ({len(trimmed_files)}/{len(csv_files)} files)")
        
        # Step 3: Create pre-merge files (sequential)
        premerge_files = []
        
        for trimmed_file in trimmed_files:
            premerge_file = self.create_premerge_file(Path(trimmed_file))
            if premerge_file:
                premerge_files.append(premerge_file)
        
        if not premerge_files:
            self.thread_print(f"SPEC {spec_num}: No pre-merge files created")
            return None
        
        self.thread_print(f"SPEC {spec_num}: ✅ Pre-merge CSVs completed ({len(premerge_files)}/{len(trimmed_files)} files)")
        
        # Step 4: Merge all files (sequential)
        merged_file = self.merge_workload_files(spec_num, premerge_files)
        
        end_time = time.time()
        duration = end_time - start_time
        
        if merged_file:
            self.thread_print(f"SPEC {spec_num}: ✅ Final CSV completed in {duration:.1f}s")
        else:
            self.thread_print(f"SPEC {spec_num}: ❌ Failed after {duration:.1f}s")
        
        return merged_file
    
    def process_all_workloads(self):
        """Process all SPEC workloads in parallel."""
        # Find all unique SPEC numbers
        out_files = list(self.raw_data_dir.glob("cpu_0_3.0GHz_spec_*.out"))
        spec_numbers = set()
        
        for out_file in out_files:
            filename = out_file.name
            if 'spec_' in filename:
                match = re.search(r'spec_(\d+)', filename)
                if match:
                    spec_num = match.group(1)
                    spec_numbers.add(spec_num)
        
        spec_numbers = sorted(spec_numbers)
        self.thread_print(f"Found SPEC workloads: {spec_numbers}")
        self.thread_print(f"Using {self.max_workers} parallel workers")
        
        results = {}
        
        # Process workloads in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_spec = {executor.submit(self.process_workload, spec_num): spec_num 
                             for spec_num in spec_numbers}
            
            for future in as_completed(future_to_spec):
                spec_num = future_to_spec[future]
                try:
                    merged_file = future.result()
                    results[spec_num] = merged_file
                except Exception as e:
                    self.thread_print(f"SPEC {spec_num} generated an exception: {e}")
                    results[spec_num] = None
        
        # Print summary
        self.thread_print(f"\n{'='*60}")
        self.thread_print("FINAL PIPELINE SUMMARY")
        self.thread_print(f"{'='*60}")
        
        for spec_num in sorted(spec_numbers):
            status = "SUCCESS" if results[spec_num] else "FAILED"
            self.thread_print(f"SPEC {spec_num}: {status}")
        
        return results

def main():
    base_dir = "/home/meb4744/PerfCount/data/arm_server"
    # Use all available CPU cores, but cap at 32 to avoid overwhelming the system
    max_workers = min(os.cpu_count(), 32)
    pipeline = FixedSPECPipeline(base_dir, max_workers=max_workers)
    
    pipeline.setup_directories()
    results = pipeline.process_all_workloads()
    
    return results

if __name__ == "__main__":
    max_workers = min(os.cpu_count(), 32)
    print(f"Starting fixed SPEC pipeline with {max_workers} workers (out of {os.cpu_count()} available cores)...")
    main()

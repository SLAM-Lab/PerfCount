#!/usr/bin/env python3
"""
Analyze instruction count variance across performance counter groups for each workload.
Compares trimmed_csvs and premerge_csvs to verify consistency.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

def extract_workload_info(filename):
    """Extract workload name, frequency, and counter group from filename"""
    # Pattern: cpu_0_FREQ_TYPE_WORKLOAD_10000000_COPY[_REDUNDANCY]_suffix.csv
    # Example: cpu_0_1.5GHz_dacapo_avrora_10000000_0_trimmed.csv
    # Example: cpu_0_3.0GHz_spec_500_10000000_0_1_trimmed.csv
    
    parts = filename.replace('_trimmed.csv', '').replace('_premerge.csv', '').split('_')
    
    # Find frequency
    freq = None
    freq_idx = -1
    for i, part in enumerate(parts):
        if 'GHz' in part:
            freq = part
            freq_idx = i
            break
    
    if freq is None or freq_idx < 2:
        return None
    
    # Extract workload type and ID
    if freq_idx + 2 < len(parts):
        workload_type = parts[freq_idx + 1]  # dacapo or spec
        workload_id = parts[freq_idx + 2]     # workload name or number
        
        # Find counter group number (last number before suffix)
        counter_group = parts[-1] if parts[-1].isdigit() else parts[-2] if len(parts) > 1 and parts[-2].isdigit() else "0"
        
        return {
            'workload': f"{workload_type}_{workload_id}",
            'frequency': freq,
            'counter_group': counter_group,
            'full_key': f"{workload_type}_{workload_id}_{freq}"
        }
    
    return None

def process_single_file(csv_file):
    """Process a single CSV file and extract instruction count"""
    try:
        info = extract_workload_info(csv_file.name)
        if not info:
            return None
        
        # Read the file
        df = pd.read_csv(csv_file)
        
        # Find instruction count
        instruction_count = 0
        
        # Check for instructions column or event column
        if 'instructions:pp:' in df.columns:
            # Pivoted format (premerge)
            instruction_count = df['instructions:pp:'].sum()
        elif 'event' in df.columns and 'count' in df.columns:
            # Un-pivoted format (trimmed)
            inst_data = df[df['event'] == 'instructions:pp:']
            if not inst_data.empty:
                # Convert count to numeric
                inst_data_copy = inst_data.copy()
                inst_data_copy['count_numeric'] = pd.to_numeric(inst_data_copy['count'], errors='coerce')
                instruction_count = inst_data_copy['count_numeric'].sum()
        
        return {
            'workload_key': info['full_key'],
            'counter_group': info['counter_group'],
            'instructions': instruction_count,
            'filename': csv_file.name
        }
        
    except Exception as e:
        return None

def analyze_directory(directory, directory_name, max_workers=None):
    """Analyze instruction counts in a directory using parallel processing"""
    print(f"\n{'='*80}")
    print(f"ANALYZING {directory_name.upper()}")
    print(f"{'='*80}\n")
    
    csv_files = list(directory.glob("*.csv"))
    total_files = len(csv_files)
    print(f" Found {total_files} files to analyze")
    
    if max_workers is None:
        max_workers = mp.cpu_count()
    
    print(f" Processing with {max_workers} parallel workers...\n")
    
    # Process files in parallel
    workload_data = defaultdict(lambda: defaultdict(dict))
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_file, f): f for f in csv_files}
        
        completed = 0
        for future in as_completed(futures):
            result = future.result()
            completed += 1
            
            if completed % 100 == 0:
                print(f"  Processed {completed}/{total_files} files...")
            
            if result:
                workload_data[result['workload_key']][result['counter_group']] = {
                    'instructions': result['instructions'],
                    'filename': result['filename']
                }
    
    print(f"\n Processed {total_files} files")
    
    # Analyze variance for each workload
    print(f"\n{'='*80}")
    print(f"VARIANCE ANALYSIS")
    print(f"{'='*80}\n")
    
    results = []
    
    for workload_key in sorted(workload_data.keys()):
        counter_groups = workload_data[workload_key]
        
        # Extract instruction counts
        instruction_counts = []
        group_nums = []
        
        for group_num in sorted(counter_groups.keys(), key=lambda x: int(x) if x.isdigit() else 0):
            inst_count = counter_groups[group_num]['instructions']
            if inst_count > 0:  # Only include non-zero counts
                instruction_counts.append(inst_count)
                group_nums.append(group_num)
        
        if len(instruction_counts) < 2:
            continue  # Need at least 2 samples
        
        # Calculate statistics
        counts_array = np.array(instruction_counts)
        mean_inst = np.mean(counts_array)
        std_inst = np.std(counts_array)
        min_inst = np.min(counts_array)
        max_inst = np.max(counts_array)
        range_inst = max_inst - min_inst
        
        # Coefficient of Variation
        cv_percent = (std_inst / mean_inst * 100) if mean_inst > 0 else 0
        
        # Range percentage
        range_percent = (range_inst / mean_inst * 100) if mean_inst > 0 else 0
        
        # Consistency classification
        if cv_percent < 0.1:
            consistency = "Excellent"
        elif cv_percent < 1.0:
            consistency = "Good"
        else:
            consistency = "Poor"
        
        result = {
            'workload': workload_key,
            'num_groups': len(instruction_counts),
            'mean_instructions': mean_inst,
            'std_instructions': std_inst,
            'min_instructions': min_inst,
            'max_instructions': max_inst,
            'range_instructions': range_inst,
            'cv_percent': cv_percent,
            'range_percent': range_percent,
            'consistency': consistency
        }
        results.append(result)
        
        # Print summary for this workload
        status = "" if cv_percent < 1.0 else "" if cv_percent < 5.0 else ""
        print(f"{status} {workload_key:<30} Groups: {len(instruction_counts):<2}  CV: {cv_percent:>6.3f}%  Range: {range_percent:>6.2f}%")
    
    # Overall statistics
    print(f"\n{'='*80}")
    print(f"SUMMARY STATISTICS - {directory_name.upper()}")
    print(f"{'='*80}\n")
    
    if results:
        results_df = pd.DataFrame(results)
        
        excellent_count = len(results_df[results_df['cv_percent'] < 0.1])
        good_count = len(results_df[(results_df['cv_percent'] >= 0.1) & (results_df['cv_percent'] < 1.0)])
        poor_count = len(results_df[results_df['cv_percent'] >= 1.0])
        
        print(f" Total workloads analyzed: {len(results_df)}")
        print(f" Excellent (CV < 0.1%): {excellent_count}")
        print(f" Good (0.1% ≤ CV < 1.0%): {good_count}")
        print(f" Poor (CV ≥ 1.0%): {poor_count}")
        print(f"\n Mean CV: {results_df['cv_percent'].mean():.3f}%")
        print(f" Median CV: {results_df['cv_percent'].median():.3f}%")
        print(f" Best: {results_df.loc[results_df['cv_percent'].idxmin(), 'workload']} ({results_df['cv_percent'].min():.3f}%)")
        print(f"  Worst: {results_df.loc[results_df['cv_percent'].idxmax(), 'workload']} ({results_df['cv_percent'].max():.3f}%)")
        
        return results_df
    else:
        print(" No results to analyze")
        return None

def compare_trimmed_vs_premerge(trimmed_results, premerge_results):
    """Compare results between trimmed and premerge directories"""
    print(f"\n{'='*80}")
    print(f"COMPARING TRIMMED VS PREMERGE CONSISTENCY")
    print(f"{'='*80}\n")
    
    if trimmed_results is None or premerge_results is None:
        print(" Cannot compare - missing results")
        return
    
    # Merge the two result sets
    comparison = pd.merge(
        trimmed_results[['workload', 'mean_instructions', 'cv_percent']],
        premerge_results[['workload', 'mean_instructions', 'cv_percent']],
        on='workload',
        suffixes=('_trimmed', '_premerge')
    )
    
    # Calculate differences
    comparison['mean_diff_percent'] = abs(comparison['mean_instructions_trimmed'] - comparison['mean_instructions_premerge']) / comparison['mean_instructions_trimmed'] * 100
    comparison['cv_diff'] = abs(comparison['cv_percent_trimmed'] - comparison['cv_percent_premerge'])
    
    # Check for matches
    comparison['means_match'] = comparison['mean_diff_percent'] < 0.01  # Within 0.01%
    comparison['cv_match'] = comparison['cv_diff'] < 0.01  # Within 0.01 percentage points
    
    matches = comparison['means_match'].sum()
    total = len(comparison)
    
    print(f" Workloads compared: {total}")
    print(f" Mean instruction counts match: {matches}/{total} ({matches/total*100:.1f}%)")
    print(f" CV values match: {comparison['cv_match'].sum()}/{total}")
    
    # Show any mismatches
    mismatches = comparison[~comparison['means_match']]
    if not mismatches.empty:
        print(f"\n  MISMATCHES (mean difference > 0.01%):")
        for _, row in mismatches.iterrows():
            print(f"  {row['workload']}: {row['mean_diff_percent']:.3f}% difference")
    else:
        print(f"\n All workloads have matching instruction counts between trimmed and premerge!")
    
    return comparison

def analyze_sample_counts(csv_file):
    """Analyze sample counts in a single premerge CSV file"""
    try:
        info = extract_workload_info(csv_file.name)
        if not info:
            return None
        
        # Read the file
        df = pd.read_csv(csv_file)
        
        # Get sample count (number of rows)
        sample_count = len(df)
        
        return {
            'workload_key': info['full_key'],
            'counter_group': info['counter_group'],
            'sample_count': sample_count,
            'filename': csv_file.name
        }
        
    except Exception as e:
        return None

def analyze_sample_count_variance(directory, max_workers=None):
    """Analyze sample count variance across counter groups"""
    print(f"\n{'='*80}")
    print(f"SAMPLE COUNT VARIANCE ANALYSIS")
    print(f"{'='*80}\n")
    
    csv_files = list(directory.glob("*.csv"))
    total_files = len(csv_files)
    print(f" Found {total_files} files to analyze")
    
    if max_workers is None:
        max_workers = mp.cpu_count()
    
    print(f" Processing with {max_workers} parallel workers...\n")
    
    # Process files in parallel
    workload_samples = defaultdict(lambda: defaultdict(int))
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(analyze_sample_counts, f): f for f in csv_files}
        
        completed = 0
        for future in as_completed(futures):
            result = future.result()
            completed += 1
            
            if completed % 100 == 0:
                print(f"  Processed {completed}/{total_files} files...")
            
            if result:
                workload_samples[result['workload_key']][result['counter_group']] = result['sample_count']
    
    print(f"\n Processed {total_files} files")
    
    # Analyze variance for each workload
    print(f"\n{'='*80}")
    print(f"SAMPLE COUNT VARIANCE BY WORKLOAD")
    print(f"{'='*80}\n")
    
    results = []
    
    for workload_key in sorted(workload_samples.keys()):
        counter_groups = workload_samples[workload_key]
        
        # Extract sample counts
        sample_counts = []
        group_nums = []
        
        for group_num in sorted(counter_groups.keys(), key=lambda x: int(x) if x.isdigit() else 0):
            count = counter_groups[group_num]
            if count > 0:
                sample_counts.append(count)
                group_nums.append(group_num)
        
        if len(sample_counts) < 2:
            continue
        
        # Calculate statistics
        counts_array = np.array(sample_counts)
        mean_samples = np.mean(counts_array)
        std_samples = np.std(counts_array)
        min_samples = np.min(counts_array)
        max_samples = np.max(counts_array)
        range_samples = max_samples - min_samples
        
        # Coefficient of Variation
        cv_percent = (std_samples / mean_samples * 100) if mean_samples > 0 else 0
        
        # Range percentage
        range_percent = (range_samples / mean_samples * 100) if mean_samples > 0 else 0
        
        # Consistency classification
        if cv_percent < 0.1:
            consistency = "Excellent"
        elif cv_percent < 1.0:
            consistency = "Good"
        else:
            consistency = "Poor"
        
        result = {
            'workload': workload_key,
            'num_groups': len(sample_counts),
            'mean_samples': mean_samples,
            'std_samples': std_samples,
            'min_samples': min_samples,
            'max_samples': max_samples,
            'range_samples': range_samples,
            'cv_percent': cv_percent,
            'range_percent': range_percent,
            'consistency': consistency
        }
        results.append(result)
        
        # Print summary for this workload
        status = "" if cv_percent < 1.0 else "" if cv_percent < 5.0 else ""
        print(f"{status} {workload_key:<35} Groups: {len(sample_counts):<2}  CV: {cv_percent:>6.3f}%  Mean: {int(mean_samples):>8,}  Range: {range_percent:>6.2f}%")
    
    # Overall statistics
    print(f"\n{'='*80}")
    print(f"SUMMARY STATISTICS - SAMPLE COUNTS")
    print(f"{'='*80}\n")
    
    if results:
        results_df = pd.DataFrame(results)
        
        excellent_count = len(results_df[results_df['cv_percent'] < 0.1])
        good_count = len(results_df[(results_df['cv_percent'] >= 0.1) & (results_df['cv_percent'] < 1.0)])
        poor_count = len(results_df[results_df['cv_percent'] >= 1.0])
        
        print(f" Total workloads analyzed: {len(results_df)}")
        print(f" Excellent (CV < 0.1%): {excellent_count}")
        print(f" Good (0.1% ≤ CV < 1.0%): {good_count}")
        print(f" Poor (CV ≥ 1.0%): {poor_count}")
        print(f"\n Mean CV: {results_df['cv_percent'].mean():.3f}%")
        print(f" Median CV: {results_df['cv_percent'].median():.3f}%")
        
        if len(results_df) > 0:
            print(f" Best: {results_df.loc[results_df['cv_percent'].idxmin(), 'workload']} ({results_df['cv_percent'].min():.3f}%)")
            print(f"  Worst: {results_df.loc[results_df['cv_percent'].idxmax(), 'workload']} ({results_df['cv_percent'].max():.3f}%)")
        
        return results_df
    else:
        print(" No results to analyze")
        return None

def analyze_per_sample_instructions(csv_file):
    """Analyze instructions per sample in a single trimmed CSV file"""
    try:
        info = extract_workload_info(csv_file.name)
        if not info:
            return None
        
        # Read the file
        df = pd.read_csv(csv_file)
        
        # Filter for instruction events
        if 'event' in df.columns and 'count' in df.columns:
            inst_data = df[df['event'] == 'instructions:pp:'].copy()
            
            if inst_data.empty:
                return None
            
            # Convert count to numeric
            inst_data['count_numeric'] = pd.to_numeric(inst_data['count'], errors='coerce')
            
            # Calculate statistics on instruction counts per sample
            instructions_per_sample = inst_data['count_numeric'].dropna()
            
            if len(instructions_per_sample) == 0:
                return None
            
            return {
                'workload_key': info['full_key'],
                'counter_group': info['counter_group'],
                'min_per_sample': instructions_per_sample.min(),
                'max_per_sample': instructions_per_sample.max(),
                'mean_per_sample': instructions_per_sample.mean(),
                'std_per_sample': instructions_per_sample.std(),
                'num_samples': len(instructions_per_sample)
            }
        
        return None
        
    except Exception as e:
        return None

def analyze_per_sample_variance(directory, max_workers=None):
    """Analyze per-sample instruction count variance across counter groups"""
    print(f"\n{'='*80}")
    print(f"PER-SAMPLE INSTRUCTION COUNT ANALYSIS")
    print(f"{'='*80}\n")
    
    csv_files = list(directory.glob("*.csv"))
    total_files = len(csv_files)
    print(f" Found {total_files} files to analyze")
    
    if max_workers is None:
        max_workers = mp.cpu_count()
    
    print(f" Processing with {max_workers} parallel workers...\n")
    
    # Process files in parallel
    workload_per_sample = defaultdict(list)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(analyze_per_sample_instructions, f): f for f in csv_files}
        
        completed = 0
        for future in as_completed(futures):
            result = future.result()
            completed += 1
            
            if completed % 100 == 0:
                print(f"  Processed {completed}/{total_files} files...")
            
            if result:
                workload_per_sample[result['workload_key']].append(result)
    
    print(f"\n Processed {total_files} files")
    
    # Analyze variance for each workload
    print(f"\n{'='*80}")
    print(f"PER-SAMPLE STATISTICS BY WORKLOAD")
    print(f"{'='*80}\n")
    
    results = []
    
    for workload_key in sorted(workload_per_sample.keys()):
        groups_data = workload_per_sample[workload_key]
        
        if len(groups_data) < 2:
            continue
        
        # Aggregate statistics across all counter groups
        all_mins = [g['min_per_sample'] for g in groups_data]
        all_maxs = [g['max_per_sample'] for g in groups_data]
        all_means = [g['mean_per_sample'] for g in groups_data]
        all_stds = [g['std_per_sample'] for g in groups_data]
        
        # Calculate variance in the statistics
        min_mean = np.mean(all_mins)
        min_std = np.std(all_mins)
        min_cv = (min_std / min_mean * 100) if min_mean > 0 else 0
        
        max_mean = np.mean(all_maxs)
        max_std = np.std(all_maxs)
        max_cv = (max_std / max_mean * 100) if max_mean > 0 else 0
        
        mean_mean = np.mean(all_means)
        mean_std = np.std(all_means)
        mean_cv = (mean_std / mean_mean * 100) if mean_mean > 0 else 0
        
        std_mean = np.mean(all_stds)
        std_std = np.std(all_stds)
        std_cv = (std_std / std_mean * 100) if std_mean > 0 else 0
        
        result = {
            'workload': workload_key,
            'num_groups': len(groups_data),
            'min_instructions_mean': min_mean,
            'min_instructions_std': min_std,
            'min_instructions_cv': min_cv,
            'max_instructions_mean': max_mean,
            'max_instructions_std': max_std,
            'max_instructions_cv': max_cv,
            'mean_instructions_mean': mean_mean,
            'mean_instructions_std': mean_std,
            'mean_instructions_cv': mean_cv,
            'std_instructions_mean': std_mean,
            'std_instructions_std': std_std,
            'std_instructions_cv': std_cv
        }
        results.append(result)
        
        # Print summary
        status = "" if mean_cv < 1.0 else ""
        print(f"{status} {workload_key:<35} Groups: {len(groups_data):<2}  Mean CV: {mean_cv:>6.3f}%  "
              f"Mean/Sample: {int(mean_mean):>10,}  Range: {int(min_mean):>10,} - {int(max_mean):>10,}")
    
    # Overall statistics
    print(f"\n{'='*80}")
    print(f"SUMMARY STATISTICS - PER-SAMPLE INSTRUCTIONS")
    print(f"{'='*80}\n")
    
    if results:
        results_df = pd.DataFrame(results)
        
        print(f" Total workloads analyzed: {len(results_df)}")
        print(f"\n Mean instruction statistics:")
        print(f"  Mean CV of mean: {results_df['mean_instructions_cv'].mean():.3f}%")
        print(f"  Mean CV of min:  {results_df['min_instructions_cv'].mean():.3f}%")
        print(f"  Mean CV of max:  {results_df['max_instructions_cv'].mean():.3f}%")
        print(f"  Mean CV of std:  {results_df['std_instructions_cv'].mean():.3f}%")
        
        print(f"\n Most consistent mean per sample: {results_df.loc[results_df['mean_instructions_cv'].idxmin(), 'workload']} (CV: {results_df['mean_instructions_cv'].min():.3f}%)")
        print(f"  Least consistent mean per sample: {results_df.loc[results_df['mean_instructions_cv'].idxmax(), 'workload']} (CV: {results_df['mean_instructions_cv'].max():.3f}%)")
        
        return results_df
    else:
        print(" No results to analyze")
        return None

def main():
    base_dir = Path("../../../data/arm_server")
    trimmed_dir = base_dir / "trimmed_csvs"
    premerge_dir = base_dir / "premerge_csvs"
    
    print("="*80)
    print("WORKLOAD INSTRUCTION COUNT VARIANCE ANALYSIS")
    print("="*80)
    
    # Analyze trimmed_csvs
    trimmed_results = analyze_directory(trimmed_dir, "trimmed_csvs")
    
    # Analyze premerge_csvs
    premerge_results = analyze_directory(premerge_dir, "premerge_csvs")
    
    # Compare the two
    if trimmed_results is not None and premerge_results is not None:
        comparison = compare_trimmed_vs_premerge(trimmed_results, premerge_results)
        
        # Save results
        output_dir = base_dir
        trimmed_results.to_csv(output_dir / "trimmed_variance_analysis.csv", index=False)
        premerge_results.to_csv(output_dir / "premerge_variance_analysis.csv", index=False)
        if comparison is not None:
            comparison.to_csv(output_dir / "trimmed_vs_premerge_comparison.csv", index=False)
        
        print(f"\n Results saved to:")
        print(f"  - {output_dir}/trimmed_variance_analysis.csv")
        print(f"  - {output_dir}/premerge_variance_analysis.csv")
        print(f"  - {output_dir}/trimmed_vs_premerge_comparison.csv")
    
    # BARRIER: Wait for instruction count analysis to complete
    print(f"\n{'='*80}")
    print(" BARRIER: Instruction count analysis complete")
    print(f"{'='*80}")
    
    # Now analyze sample count variance
    sample_results = analyze_sample_count_variance(premerge_dir)
    
    if sample_results is not None:
        sample_results.to_csv(output_dir / "sample_count_variance_analysis.csv", index=False)
        print(f"\n Sample count results saved to:")
        print(f"  - {output_dir}/sample_count_variance_analysis.csv")
    
    # BARRIER: Wait for sample count analysis to complete
    print(f"\n{'='*80}")
    print(" BARRIER: Sample count analysis complete")
    print(f"{'='*80}")
    
    # Now analyze per-sample instruction variance
    per_sample_results = analyze_per_sample_variance(trimmed_dir)
    
    if per_sample_results is not None:
        per_sample_results.to_csv(output_dir / "per_sample_instruction_analysis.csv", index=False)
        print(f"\n Per-sample instruction results saved to:")
        print(f"  - {output_dir}/per_sample_instruction_analysis.csv")

if __name__ == "__main__":
    main()


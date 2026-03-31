#!/usr/bin/env python3
"""
Script to convert ARM server data to the format expected by phase-aware PMU forecasting scripts.
"""

import pandas as pd
import os
import sys
from pathlib import Path

def convert_arm_data_to_forecasting_format(input_file, output_file):
    """
    Convert ARM server CSV data to the format expected by the forecasting scripts.
    
    ARM server columns -> Expected format:
    - instructionspp -> instructions:u
    - cpu-cycles -> cpu-cycles:u  
    - branch-loads -> br_inst_retired.all_branches_pebs:u
    - branch-misses -> br_misp_retired.all_branches_pebs:u
    """
    
    # Read the ARM server data
    df = pd.read_csv(input_file)
    
    # Create the converted dataframe with expected column names
    converted_df = pd.DataFrame()
    
    # Map the columns to the expected format
    converted_df['instructions:u'] = df['instructionspp']
    converted_df['cpu-cycles:u'] = df['cpu-cycles']
    converted_df['br_inst_retired.all_branches_pebs:u'] = df['branch-loads']
    converted_df['br_misp_retired.all_branches_pebs:u'] = df['branch-misses']
    
    # Save the converted data
    converted_df.to_csv(output_file, index=False)
    print(f"Converted {input_file} -> {output_file}")
    print(f"Shape: {converted_df.shape}")
    print(f"Columns: {list(converted_df.columns)}")

def main():
    # Create the arm_server dataset directory
    arm_server_dir = Path("/home/meb4744/phase-aware_pmu_forecasting/Data/arm_server")
    arm_server_dir.mkdir(parents=True, exist_ok=True)
    
    # Source directory with ARM server data
    source_dir = Path("/home/meb4744/PerfCount/data/arm_server/training_csvs")
    
    # Convert each CSV file
    for csv_file in source_dir.glob("*.csv"):
        output_file = arm_server_dir / csv_file.name
        convert_arm_data_to_forecasting_format(csv_file, output_file)

if __name__ == "__main__":
    main()


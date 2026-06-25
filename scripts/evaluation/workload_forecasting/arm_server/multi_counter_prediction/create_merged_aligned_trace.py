import os
import glob
import pandas as pd

# --- CONFIGURATION ---
# Set the directory where your aligned CSVs are stored
DATA_DIR = os.path.expanduser("~/PerfCount/processed_data_10M/arm_server")
OUTPUT_FILE = os.path.join(DATA_DIR, "combined_aligned_traces.csv")

def concatenate_traces():
    search_pattern = os.path.join(DATA_DIR, "**", "*aligned*.csv")
    csv_files = glob.glob(search_pattern, recursive=True)
    
    if not csv_files:
        print(f"No files found matching {search_pattern}")
        return
        
    print(f"Found {len(csv_files)} files. Starting concatenation...")
    
    df_list = []
    total_rows = 0
    
    for file in csv_files:
        try:
            # Read the CSV
            df = pd.read_csv(file)
            
            # Extract the filename to keep track of where the data came from
            filename = os.path.basename(file)
            df['source_file'] = filename
            
            df_list.append(df)
            total_rows += len(df)
            print(f"Loaded: {filename:<50} | Rows: {len(df)}")
            
        except Exception as e:
            print(f"Error reading {file}: {e}")
            
    if df_list:
        print("\nConcatenating all dataframes (this might take a moment)...")
        # Combine all the dataframes vertically
        combined_df = pd.concat(df_list, ignore_index=True)
        
        print(f"Saving combined data to {OUTPUT_FILE}...")
        combined_df.to_csv(OUTPUT_FILE, index=False)
        print(f"Successfully saved {len(combined_df)} total rows to {OUTPUT_FILE}!")
    else:
        print("No data was loaded.")

if __name__ == "__main__":
    concatenate_traces()
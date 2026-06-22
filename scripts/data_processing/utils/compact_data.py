import os
import glob
import pandas as pd
import argparse

def compact_csv(input_path, output_path, block_size):
    try:
        df = pd.read_csv(input_path)
        
        # We don't want to sum the old sample_index, so we drop it if it exists.
        # We will create a fresh one for the 100M blocks.
        if 'sample_index' in df.columns:
            df = df.drop(columns=['sample_index'])
            
        # Create a grouping key where every block_size rows get the same integer id
        # integer division by block_size: 0//10 = 0, 9//10 = 0, 10//10 = 1, etc.
        df['group'] = df.index // block_size
        
        # Group and sum all hardware event columns
        df_compacted = df.groupby('group').sum().reset_index()
        
        # Rename the 'group' column to 'sample_index' for the new 100M blocks
        df_compacted = df_compacted.rename(columns={'group': 'sample_index'})
        
        # Force all data to be strictly integers (fill any missing values with 0 first to prevent float casting)
        df_compacted = df_compacted.fillna(0).astype(int)
        
        # Save the new 100M block CSV
        df_compacted.to_csv(output_path, index=False)
        return True
    
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Compact 10M aligned CSVs into 100M block CSVs.")
    parser.add_argument("--in_dir", default=".", help="Directory containing the original 10M aligned CSVs")
    parser.add_argument("--out_dir", default="compacted_100M", help="Directory to save the compacted CSVs")
    parser.add_argument("--block_size", type=int, default=10, help="Number of rows to aggregate (default: 10)")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    files = glob.glob(os.path.join(args.in_dir, "**/aligned_*.csv"), recursive=True)

    if not files:
        print(f"No matching 'aligned_*.csv' files found in '{args.in_dir}'.")
        return

    print(f"Found {len(files)} aligned traces. Compacting (block size = {args.block_size})...")

    success_count = 0
    for f in files:
        rel_path = os.path.relpath(f, args.in_dir)
        out_path = os.path.join(args.out_dir, rel_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        if compact_csv(f, out_path, args.block_size):
            success_count += 1

    print(f"Successfully compacted {success_count} / {len(files)} files into '{args.out_dir}'.")
    print("All output data has been strictly formatted as integers.")

if __name__ == "__main__":
    main()
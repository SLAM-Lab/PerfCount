import os
import argparse
import pandas as pd
import numpy as np
import glob
import re
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor

# ==========================================
# 1. FEATURE ENGINEERING
# ==========================================
def extract_features_dynamic(df, suffix=""):
    X = pd.DataFrame()
    
    # Base metrics for rate calculations
    col_instr = f"instructions{suffix}"
    col_cycles = f"cpu_cycles{suffix}"

    if col_instr not in df.columns or col_cycles not in df.columns:
        return pd.DataFrame()

    vec_instr = df[col_instr].replace(0, np.nan)
    vec_cycles = df[col_cycles].replace(0, np.nan)

    # Base performance metrics
    X["ipc"] = df[col_instr] / (vec_cycles.fillna(1e-9))
    X["cpi"] = df[col_cycles] / (vec_instr.fillna(1e-9))

    # Strict whitelist of columns for ARM Desktop
    # Excludes: sample_index, cpu_cycles, instructions, cpu_clock, task_clock
    hardware_events = [
        "branch_load_misses", "branch_loads", "branch_misses", "branches",
        "bx_stall", "cache_misses", "cache_references", "decode_stall",
        "dispatch_stall", "dtlb_load_misses", "dtlb_loads", "dtlb_walk",
        "faults", "fx_stall", "itlb_load_misses", "itlb_loads", "itlb_walk",
        "ixa_stall", "ixb_stall", "l1_dcache_load_misses", "l1_dcache_loads",
        "l1_icache_load_misses", "l1_icache_loads", "l1d_cache", "l1i_cache",
        "lx_stall", "mem_access", "mem_access_rd", "mem_access_wr",
        "minor_faults", "page_faults", "sx_stall"
    ]

    # Convert raw counts to stable rates
    for base_event in hardware_events:
        col = f"{base_event}{suffix}"
        if col in df.columns:
            X[f"{base_event}_per_instr"] = df[col] / (vec_instr.fillna(1e-9))
            X[f"{base_event}_per_cycle"] = df[col] / (vec_cycles.fillna(1e-9))

    return X.replace([np.inf, -np.inf], 0).fillna(0)

# ==========================================
# 2. WORKER FUNCTION
# ==========================================
def process_fold(test_bench, train_dfs, test_df, freq_ratio, target_key, out_dir):
    try:
        train_full = pd.concat(train_dfs, ignore_index=True)
        
        X_train_full = extract_features_dynamic(train_full, suffix="_src")
        X_test = extract_features_dynamic(test_df, suffix="_src")

        if X_train_full.empty:
            return None 
            
        for c in X_train_full.columns:
            if c not in X_test.columns: X_test[c] = 0
        X_test = X_test[X_train_full.columns]

        src_clean = train_full['source_val'].replace(0, np.nan).fillna(1e-9)
        train_ratios = train_full['target_y'] / src_clean
        y_tr_ratio = np.clip(train_ratios, 0.5, 3.0) 

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train_full, y_tr_ratio, test_size=0.2, random_state=42
        )

        model = CatBoostRegressor(
            iterations=1000, 
            depth=6, 
            learning_rate=0.05, 
            loss_function='RMSE',
            verbose=False,
            allow_writing_files=False,
            thread_count=4
        )
        
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=100)
        
        models_dir = os.path.join(out_dir, "models")
        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, f"model_fold_{test_bench}.cbm")
        model.save_model(model_path)
        
        diag_dir = os.path.join(out_dir, "diagnostics")
        os.makedirs(diag_dir, exist_ok=True)
        
        importances = model.get_feature_importance()
        feat_imp_dict = dict(zip(X_train_full.columns, importances))
        
        feat_imp = pd.Series(importances, index=X_train_full.columns).sort_values(ascending=False)
        
        plt.figure(figsize=(10, 6))
        feat_imp.head(10).plot(kind='bar')
        plt.title(f'Top 10 Features - {test_bench}')
        plt.ylabel('Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(diag_dir, f"{test_bench}_feat_imp.png"))
        plt.close()

        plt.figure(figsize=(8, 6))
        plt.scatter(X_train_full['cpi'], y_tr_ratio, alpha=0.1, s=2)
        plt.title(f'Ground Truth: Source CPI vs Target Scaling Ratio\n(Without Test Bench: {test_bench})')
        plt.xlabel('Source CPI (Cycles Per Instruction)')
        plt.ylabel('Actual Cycle Scaling Ratio (Target / Source)')
        plt.axhline(1.0, color='r', linestyle='--', label='Compute Bound (1.0x)')
        plt.axhline(freq_ratio, color='g', linestyle='--', label=f'Memory Bound ({freq_ratio}x)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(diag_dir, f"{test_bench}_scatter.png"))
        plt.close()

        pred_ratio = model.predict(X_test)
        src_test_val = test_df['source_val']
        y_test = test_df['target_y']
        
        pred_ml = pred_ratio * src_test_val
        pred_copy = src_test_val
        pred_scale = src_test_val * freq_ratio

        mask = y_test > 0
        def get_mape(y, p): return mean_absolute_percentage_error(y[mask], p[mask]) * 100

        return {
            "bench": test_bench,
            "mape_ml": get_mape(y_test, pred_ml),
            "mape_copy": get_mape(y_test, pred_copy),
            "mape_scale": get_mape(y_test, pred_scale),
            "feat_imp": feat_imp_dict  # Passed back to orchestrator
        }
    except Exception as e:
        return None

# ==========================================
# 3. ORCHESTRATION
# ==========================================
def run_comparison(data_map, src_freq, tgt_freq, target_key, out_dir, n_jobs, strict_loocv):
    benches = sorted(list(set(k[1] for k in data_map.keys() if k[0] == src_freq)))
    valid_benches = []
    bench_dfs = {}
    
    for b in benches:
        if (tgt_freq, b) in data_map:
            df_s = data_map[(src_freq, b)].copy().add_suffix('_src')
            df_t = data_map[(tgt_freq, b)].copy().add_suffix('_tgt')
            
            src_instr_col = "instructions_src"
            tgt_instr_col = "instructions_tgt"
            
            if src_instr_col in df_s.columns and tgt_instr_col in df_t.columns:
                df_s['cum_instr'] = df_s[src_instr_col].fillna(0).cumsum()
                df_t['cum_instr'] = df_t[tgt_instr_col].fillna(0).cumsum()
                
                df_s = df_s.sort_values('cum_instr')
                df_t = df_t.sort_values('cum_instr')

                try:
                    merged = pd.merge_asof(
                        df_s, df_t, 
                        on='cum_instr',
                        direction='nearest'
                    )
                except Exception as e:
                    print(f"CRASH IN WORKLOAD: {b}")
                    raise e
            else:
                min_len = min(len(df_s), len(df_t))
                merged = pd.concat([df_s.iloc[:min_len].reset_index(drop=True), 
                                  df_t.iloc[:min_len].reset_index(drop=True)], axis=1)

            tgt_col = f"{target_key}_tgt"
            src_col = f"{target_key}_src"
            
            if tgt_col in merged.columns and src_col in merged.columns:
                merged['target_y'] = merged[tgt_col]
                merged['source_val'] = merged[src_col]
                
                merged = merged[(merged['target_y'] > 0) & (merged['source_val'] > 0)]
                
                if not merged.empty:
                    bench_dfs[b] = merged
                    valid_benches.append(b)

    if len(valid_benches) < 2:
        return None

    try:
        ratio = float(tgt_freq) / float(src_freq)
    except:
        ratio = 1.0

    print(f"    Starting folds (Strict LOOCV: {strict_loocv})...")
    direction_dir = os.path.join(out_dir, f"{src_freq}to{tgt_freq}")
    os.makedirs(direction_dir, exist_ok=True)
    
    # LOOCV LOGIC
    if strict_loocv:
        # Group phases by their base workload name (ignores the cpuX and phaseY suffix for splitting)
        groups = {}
        for b in valid_benches:
            base_name = b.split('_cpu')[0]
            groups.setdefault(base_name, []).append(b)
            
        tasks = []
        for base_name, t_benches in groups.items():
            train_benches = [b for b in valid_benches if b not in t_benches]
            train_dfs = [bench_dfs[x] for x in train_benches]
            test_df = pd.concat([bench_dfs[x] for x in t_benches], ignore_index=True)
            
            tasks.append(delayed(process_fold)(base_name, train_dfs, test_df, ratio, target_key, direction_dir))
    else:
        tasks = [
            delayed(process_fold)(b, [bench_dfs[x] for x in valid_benches if x != b], bench_dfs[b], ratio, target_key, direction_dir)
            for b in valid_benches
        ]

    results = Parallel(n_jobs=n_jobs, verbose=0, backend="multiprocessing")(tasks)
    results = [r for r in results if r is not None]

    if not results:
        return None

    df_res = pd.DataFrame(results)
    
    # Process and save global average feature importance
    feat_df = pd.DataFrame(df_res['feat_imp'].tolist())
    avg_feat_imp = feat_df.mean().sort_values(ascending=False)
    
    # Save Feature Importance Plot
    plt.figure(figsize=(12, 7))
    avg_feat_imp.head(15).plot(kind='bar')
    plt.title(f'Average Feature Importance across all Folds ({src_freq}GHz to {tgt_freq}GHz)')
    plt.ylabel('Average Importance %')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(direction_dir, f"average_feature_importance_{src_freq}to{tgt_freq}.png"))
    plt.close()
    
    # Save Feature Importance CSV
    avg_feat_imp.to_csv(os.path.join(direction_dir, f"average_feature_importance_{src_freq}to{tgt_freq}.csv"), header=['Importance'])

    # Drop the dictionary column before saving the main per-fold results
    df_res = df_res.drop(columns=['feat_imp'])
    
    per_fold_csv = os.path.join(direction_dir, f"per_fold_results_{target_key}.csv")
    df_res.to_csv(per_fold_csv, index=False)
    
    avg_ml = df_res['mape_ml'].mean()
    avg_copy = df_res['mape_copy'].mean()
    avg_scale = df_res['mape_scale'].mean()
    
    print(f"    > {target_key} | ML: {avg_ml:.2f}% | Copy: {avg_copy:.2f}% | Scale: {avg_scale:.2f}%")
    
    return {
        "pair": f"{src_freq}->{tgt_freq}",
        "target": target_key,
        "ml": avg_ml,
        "copy": avg_copy,
        "scale": avg_scale
    }

def process_processor_dataset(processor_name, data_map, base_out_dir, jobs, strict_loocv):
    print(f"\n==============================================")
    print(f" RUNNING PIPELINE FOR: {processor_name}")
    print(f"==============================================")
    
    if not data_map:
        print(f"No data found for {processor_name}.")
        return

    proc_out_dir = os.path.join(base_out_dir, processor_name)
    os.makedirs(proc_out_dir, exist_ok=True)

    freqs = sorted(list(set(k[0] for k in data_map.keys())))
    
    # Explicitly predict the exact cpu_cycles column
    targets = ['cpu_cycles']
    
    summary = []
    for src in freqs:
        for tgt in freqs:
            if src == tgt: continue
            print(f"\n  --- {src}GHz -> {tgt}GHz ---")
            for t in targets:
                res = run_comparison(data_map, src, tgt, t, proc_out_dir, jobs, strict_loocv)
                if res: summary.append(res)

    if summary:
        print(f"\n{processor_name} SUMMARY:")
        df = pd.DataFrame(summary)
        print(df.to_string(index=False))
        df.to_csv(os.path.join(proc_out_dir, "summary.csv"), index=False)

def main():
    parser = argparse.ArgumentParser()
    # Updated paths specifically for arm_desktop
    parser.add_argument("--data_dir", default="../../../../processed_data/arm_desktop", help="Directory containing the aligned ARM Desktop CSVs")
    parser.add_argument("--out_dir", default="../../../../results/cross_platform/cross_frequency/arm_desktop", help="Base output directory")
    parser.add_argument("--jobs", type=int, default=os.cpu_count(), help="Number of parallel jobs")
    parser.add_argument("--strict_loocv", action="store_true", help="Group all phases of the same workload into the test set")
    args = parser.parse_args()

    datasets = {
        "arm_desktop": {},
    }

    # 1. Load ARM Desktop Data dynamically
    if os.path.exists(args.data_dir):
        aligned_files = glob.glob(os.path.join(args.data_dir, "aligned_*.csv"))
        for f in aligned_files:
            try:
                fname = os.path.basename(f)
                # Updated regex to correctly identify the cpu ID in desktop aligned files
                match = re.search(r"aligned_(?P<bench>.+)_(?P<freq>[\d\.]+)GHz_cpu(?P<cpu>\d+)_phase(?P<phase>\d+)\.csv", fname)
                
                if match:
                    bench = match.group('bench')
                    freq = match.group('freq')
                    cpu = match.group('cpu')
                    phase = match.group('phase')
                    
                    full_bench = f"{bench}_cpu{cpu}_phase{phase}"
                    
                    df = pd.read_csv(f)
                    df.columns = [c.strip() for c in df.columns]
                    datasets["arm_desktop"][(freq, full_bench)] = df
            except Exception as e:
                print(f"Failed to load {f}: {e}")

    # 2. Process
    for proc_name, data_map in datasets.items():
        process_processor_dataset(proc_name, data_map, args.out_dir, args.jobs, args.strict_loocv)

if __name__ == "__main__":
    main()
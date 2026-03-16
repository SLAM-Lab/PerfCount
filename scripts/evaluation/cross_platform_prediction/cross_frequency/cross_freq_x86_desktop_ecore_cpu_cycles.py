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
    
    col_instr = f"instructions{suffix}"
    col_cycles = f"cpu_cycles{suffix}"

    if col_instr not in df.columns or col_cycles not in df.columns:
        return pd.DataFrame()

    vec_instr = df[col_instr].replace(0, np.nan)
    vec_cycles = df[col_cycles].replace(0, np.nan)

    X["ipc"] = df[col_instr] / (vec_cycles.fillna(1e-9))
    X["cpi"] = df[col_cycles] / (vec_instr.fillna(1e-9))

    # Strict whitelist for CPU 16 (Intel E-Core) - L1 D-Cache missing
    hardware_events = [
        "branch_load_misses", "branch_loads", "branch_misses", "branches", 
        "bus_cycles", "cache_misses", "cache_references", "dtlb_load_misses", 
        "dtlb_loads", "dtlb_store_misses", "dtlb_stores", "itlb_load_misses", 
        "l1_icache_load_misses", "llc_loads", "llc_misses", "mem_stores", 
        "ref_cycles"
    ]

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

        if X_train_full.empty: return None 
            
        for c in X_train_full.columns:
            if c not in X_test.columns: X_test[c] = 0
        X_test = X_test[X_train_full.columns]

        src_clean = train_full['source_val'].replace(0, np.nan).fillna(1e-9)
        train_ratios = train_full['target_y'] / src_clean
        y_tr_ratio = np.clip(train_ratios, 0.5, 3.0) 

        X_tr, X_val, y_tr, y_val = train_test_split(X_train_full, y_tr_ratio, test_size=0.2, random_state=42)

        model = CatBoostRegressor(
            iterations=1000, depth=6, learning_rate=0.05, loss_function='RMSE',
            verbose=False, allow_writing_files=False, thread_count=4
        )
        
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=100)
        
        models_dir = os.path.join(out_dir, "models")
        os.makedirs(models_dir, exist_ok=True)
        model.save_model(os.path.join(models_dir, f"model_fold_{test_bench}.cbm"))
        
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

        pred_ratio = model.predict(X_test)
        src_test_val = test_df['source_val']
        y_test = test_df['target_y']
        
        pred_ml = pred_ratio * src_test_val
        pred_copy = src_test_val
        pred_scale = src_test_val * freq_ratio

        mask = y_test > 0
        def get_mape(y, p): return mean_absolute_percentage_error(y[mask], p[mask]) * 100

        return {
            "bench": test_bench, "mape_ml": get_mape(y_test, pred_ml),
            "mape_copy": get_mape(y_test, pred_copy), "mape_scale": get_mape(y_test, pred_scale),
            "feat_imp": feat_imp_dict
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
            
            if "instructions_src" in df_s.columns and "instructions_tgt" in df_t.columns:
                df_s['cum_instr'] = df_s["instructions_src"].fillna(0).cumsum()
                df_t['cum_instr'] = df_t["instructions_tgt"].fillna(0).cumsum()
                df_s = df_s.sort_values('cum_instr')
                df_t = df_t.sort_values('cum_instr')
                merged = pd.merge_asof(df_s, df_t, on='cum_instr', direction='nearest')
            else:
                min_len = min(len(df_s), len(df_t))
                merged = pd.concat([df_s.iloc[:min_len].reset_index(drop=True), 
                                  df_t.iloc[:min_len].reset_index(drop=True)], axis=1)

            if f"{target_key}_tgt" in merged.columns and f"{target_key}_src" in merged.columns:
                merged['target_y'] = merged[f"{target_key}_tgt"]
                merged['source_val'] = merged[f"{target_key}_src"]
                merged = merged[(merged['target_y'] > 0) & (merged['source_val'] > 0)]
                
                if not merged.empty:
                    bench_dfs[b] = merged
                    valid_benches.append(b)

    if len(valid_benches) < 2: return None

    ratio = float(tgt_freq) / float(src_freq) if float(src_freq) > 0 else 1.0

    print(f"    Starting folds (Strict LOOCV: {strict_loocv})...")
    direction_dir = os.path.join(out_dir, f"{src_freq}to{tgt_freq}")
    os.makedirs(direction_dir, exist_ok=True)
    
    if strict_loocv:
        groups = {}
        for b in valid_benches: groups.setdefault(b.split('_phase')[0], []).append(b)
        tasks = []
        for base_name, t_benches in groups.items():
            train_benches = [b for b in valid_benches if b not in t_benches]
            tasks.append(delayed(process_fold)(base_name, [bench_dfs[x] for x in train_benches], 
                                               pd.concat([bench_dfs[x] for x in t_benches], ignore_index=True), 
                                               ratio, target_key, direction_dir))
    else:
        tasks = [delayed(process_fold)(b, [bench_dfs[x] for x in valid_benches if x != b], bench_dfs[b], ratio, target_key, direction_dir) for b in valid_benches]

    results = [r for r in Parallel(n_jobs=n_jobs, verbose=0, backend="multiprocessing")(tasks) if r is not None]
    if not results: return None

    df_res = pd.DataFrame(results)
    
    avg_feat_imp = pd.DataFrame(df_res['feat_imp'].tolist()).mean().sort_values(ascending=False)
    plt.figure(figsize=(12, 7))
    avg_feat_imp.head(15).plot(kind='bar')
    plt.title(f'Average Feature Importance across all Folds ({src_freq}GHz to {tgt_freq}GHz)')
    plt.ylabel('Average Importance %')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(direction_dir, f"average_feature_importance_{src_freq}to{tgt_freq}.png"))
    plt.close()
    
    avg_feat_imp.to_csv(os.path.join(direction_dir, f"average_feature_importance_{src_freq}to{tgt_freq}.csv"), header=['Importance'])

    df_res = df_res.drop(columns=['feat_imp'])
    df_res.to_csv(os.path.join(direction_dir, f"per_fold_results_{target_key}.csv"), index=False)
    
    print(f"    > {target_key} | ML: {df_res['mape_ml'].mean():.2f}% | Copy: {df_res['mape_copy'].mean():.2f}% | Scale: {df_res['mape_scale'].mean():.2f}%")
    return {"pair": f"{src_freq}->{tgt_freq}", "target": target_key, "ml": df_res['mape_ml'].mean(), "copy": df_res['mape_copy'].mean(), "scale": df_res['mape_scale'].mean()}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../../../../processed_data/x86_desktop_heterogeneous", help="Directory containing the aligned x86 CSVs")
    parser.add_argument("--out_dir", default="../../../../results/cross_platform/cross_frequency/x86_desktop_heterogeneous/cpu16_ecore", help="Base output directory")
    parser.add_argument("--jobs", type=int, default=os.cpu_count(), help="Number of parallel jobs")
    parser.add_argument("--strict_loocv", action="store_true", help="Group all phases of the same workload into the test set")
    args = parser.parse_args()

    datasets = {"x86_cpu16": {}}

    if os.path.exists(args.data_dir):
        for f in glob.glob(os.path.join(args.data_dir, "aligned_*.csv")):
            try:
                fname = os.path.basename(f)
                match = re.search(r"aligned_(?P<bench>.+)_(?P<freq>[\d\.]+)GHz_cpu(?P<cpu>\d+)_phase(?P<phase>\d+)\.csv", fname)
                
                if match and match.group('cpu') == "16":
                    df = pd.read_csv(f)
                    df.columns = [c.strip() for c in df.columns]
                    datasets["x86_cpu16"][(match.group('freq'), f"{match.group('bench')}_phase{match.group('phase')}")] = df
            except Exception as e:
                print(f"Failed to load {f}: {e}")

    for proc_name, data_map in datasets.items():
        if not data_map: continue
        os.makedirs(args.out_dir, exist_ok=True)
        freqs = sorted(list(set(k[0] for k in data_map.keys())))
        summary = []
        for src in freqs:
            for tgt in freqs:
                if src == tgt: continue
                print(f"\n  --- CPU 16 | {src}GHz -> {tgt}GHz ---")
                res = run_comparison(data_map, src, tgt, 'cpu_cycles', args.out_dir, args.jobs, args.strict_loocv)
                if res: summary.append(res)
        if summary:
            pd.DataFrame(summary).to_csv(os.path.join(args.out_dir, "summary.csv"), index=False)

if __name__ == "__main__":
    main()
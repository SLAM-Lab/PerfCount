"""
shared_features.py
==================
Single source of truth for feature engineering, model configuration,
data alignment, LOOCV dispatch, and metrics.

All cross_* scripts import from here. The model architecture, feature
construction rules, and evaluation protocol are therefore identical
across cross_freq_arm, cross_freq_x86, cross_proc, and cross_sys.
"""

import os

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from joblib import Parallel, delayed


# =============================================================================
# 0.  COMMAND-LINE FLAG DEFINITIONS
#     Call add_feature_args(parser) in each script to attach the shared flags.
# =============================================================================

# DaCapo workloads whose total dynamic instruction count varies sharply across
# frequency/core configurations (>19% CV in repeated-run testing, vs. <3% for
# every other DaCapo workload) — tradebeans/tradesoap/h2o run for a roughly
# fixed wall-clock duration rather than a fixed amount of work, so "same
# workload" does not mean "same amount of work" once frequency or core changes.
UNSTABLE_DACAPO_WORKLOADS = ["cassandra", "tradebeans", "tradesoap", "h2o", "kafka"]


def add_feature_args(parser):
    """Attach the shared feature-toggle and model flags to an argparse parser."""
    g = parser.add_argument_group("Feature toggles")

    g.add_argument("--input_counters", nargs="+", default=None,
                   help="Restrict raw hardware counters used as features to exactly "
                        "this list (no counters are force-kept). "
                        "Default: use all available counters.")

    g.add_argument("--jobs", type=int, default=8,
                   help="Parallel joblib workers")
    g.add_argument("--strict_loocv", action="store_true", default=True,
                   help="Group all phases of the same workload into the test set (default: on)")
    g.add_argument("--no_strict_loocv", dest="strict_loocv", action="store_false")

    g.add_argument("--equal_weight", action="store_true", default=False,
                   help="Weight training samples so every workload contributes equally "
                        "regardless of trace length (1/len per sample). "
                        "Prevents long workloads from dominating the fit.")

    g.add_argument("--mode", default="loocv",
                   choices=["loocv", "general_insample", "general_temporal"],
                   help="loocv (default): each workload predicted by a model trained on "
                        "the others (generalization to unseen workloads). "
                        "general_insample: one model trained on ALL workloads, evaluated on "
                        "each (learnability ceiling). general_temporal: one model trained on "
                        "the first --temporal_frac of every workload, tested on the held-out "
                        "tail (deployable shared model).")
    g.add_argument("--temporal_frac", type=float, default=0.8,
                   help="general_temporal only: fraction of each workload's trace used for "
                        "training; the remaining tail is the test split (default 0.8).")

    g.add_argument("--exclude_unstable_dacapo", action="store_true", default=False,
                   help="Drop DaCapo workloads with high cross-configuration variance "
                        f"in total instruction count ({', '.join(UNSTABLE_DACAPO_WORKLOADS)}) "
                        "from the evaluation. No effect on SPEC workloads.")

    return parser


def filter_excluded_benchmarks(data_map, args):
    """
    Drop entries whose base workload name is in UNSTABLE_DACAPO_WORKLOADS
    when --exclude_unstable_dacapo is set. No-op otherwise.

    Parameters
    ----------
    data_map : dict[(freq_str, bench_phase_str) -> pd.DataFrame]
        Keys carry the workload name as 'bench_phaseN'; matches the
        'bench_phase_str' convention used by every cross_* loader.
    args     : argparse.Namespace  (carries --exclude_unstable_dacapo)
    """
    if not getattr(args, "exclude_unstable_dacapo", False):
        return data_map

    def base_name(bench_phase_key):
        # DaCapo aligned files are named 'aligned_dacapo_<bench>_...', so the
        # parsed bench id carries a literal 'dacapo_' prefix; SPEC bench ids
        # don't, so stripping it here is safe for both suites.
        name = bench_phase_key.split("_phase")[0]
        return name[len("dacapo_"):] if name.startswith("dacapo_") else name

    filtered = {
        k: v for k, v in data_map.items()
        if base_name(k[1]) not in UNSTABLE_DACAPO_WORKLOADS
    }
    n_dropped = len(data_map) - len(filtered)
    if n_dropped:
        print(f"  [exclude_unstable_dacapo] Dropped {n_dropped} file(s) "
              f"for workloads: {', '.join(UNSTABLE_DACAPO_WORKLOADS)}")
    return filtered


# =============================================================================
# 1.  CUSTOM CATBOOST METRIC  (Linear-space MAPE evaluated in log-space)
# =============================================================================

class LinearMAPE:
    """
    CatBoost custom eval metric.
    Trains in log-space for numerical stability; early-stopping uses
    true linear-space percentage error so validation curves are human-readable.
    """
    def get_final_error(self, error, weight):
        return error / (weight + 1e-9)

    def is_max_optimal(self):
        return False  # lower is better

    def evaluate(self, approxes, target, weight):
        preds      = np.exp(approxes[0])
        actuals    = np.exp(target)
        error_sum  = np.sum(np.abs((actuals - preds) / (actuals + 1e-9)))
        weight_sum = len(actuals)
        return error_sum, weight_sum


# =============================================================================
# 2.  MODEL FACTORY
#     Single canonical CatBoost config used everywhere.
# =============================================================================

def build_model(cat_features=None):
    """
    Return a fresh CatBoostRegressor with the canonical physics-baseline config.

    Parameters
    ----------
    cat_features : list[str] or None
        Column names that should be treated as categoricals (e.g. ['bottleneck_class']).
        Pass None or [] if no categorical features are present.

    Notes
    -----
    * loss_function  = MAE                  (symmetric; no directional bias)
    * eval_metric    = LinearMAPE()         (human-readable; drives early stopping)
    * Target         = log(ratio)           — callers must log-transform before fit()
                       and exp() after predict().
    """
    if cat_features is None:
        cat_features = []

    return CatBoostRegressor(
        iterations      = 1000,
        depth           = 5,
        learning_rate   = 0.03,
        loss_function   = "MAE",
        eval_metric     = LinearMAPE(),
        cat_features    = cat_features if cat_features else None,
        l2_leaf_reg     = 30,
        random_strength = 5,
        rsm             = 0.7,
        od_type         = "Iter",
        od_wait         = 100,
        verbose         = False,
        allow_writing_files = False,
        thread_count    = 6,
    )


# =============================================================================
# 3.  FEATURE ENGINEERING
# =============================================================================

ALWAYS_KEEP_COUNTERS = []  # no forced baselines: a top-K model uses exactly K counters


def restrict_input_counters(df, suffix, input_counters):
    """Drop raw counter columns (with the given suffix) not in
    ALWAYS_KEEP_COUNTERS or input_counters."""
    if not input_counters:
        return df
    keep = {f"{c}{suffix}" for c in ALWAYS_KEEP_COUNTERS} | \
           {f"{c}{suffix}" for c in input_counters}
    drop_cols = [c for c in df.columns if c.endswith(suffix) and c not in keep]
    return df.drop(columns=drop_cols, errors="ignore")


def build_features(df, suffix, args):
    """
    Extract raw hardware counter columns as the feature matrix.

    Parameters
    ----------
    df     : pd.DataFrame  — columns are already suffixed (e.g. 'instructions_src')
    suffix : str           — e.g. '_src' or '_tgt' or ''
    args   : argparse.Namespace  — carries input_counters filter

    Returns
    -------
    X      : pd.DataFrame  — feature matrix (never contains NaN/Inf)
    """
    non_feature = {"sample_index", "target_y", "source_val"}

    def _base(c):
        # strip the _src/_tgt suffix so the exclusion matches suffixed columns
        # (e.g. 'sample_index_src') and sample_index never leaks as a feature.
        return c[:-len(suffix)] if suffix and c.endswith(suffix) else c

    counter_cols = [c for c in df.columns if c.endswith(suffix)] if suffix else list(df.columns)
    counter_cols = [c for c in counter_cols if _base(c) not in non_feature]

    if not counter_cols:
        return pd.DataFrame(index=df.index)

    X = df[counter_cols].copy()
    if suffix:
        X.columns = [c[:-len(suffix)] for c in X.columns]

    return X.replace([np.inf, -np.inf], 0).fillna(0)


def cat_feature_names(args):
    """Return the list of categorical feature names active under current args."""
    return []


# =============================================================================
# 4.  DATA ALIGNMENT
#     Merges source and target DataFrames on cumulative instruction count.
# =============================================================================

def merge_by_cumulative_instructions(df_src, df_tgt,
                                     src_instr_col="instructions",
                                     tgt_instr_col="instructions"):
    """
    Align two DataFrames (source and target) on cumulative instruction count
    using an asof merge (nearest neighbour in instruction space).

    Returns a merged DataFrame or an empty DataFrame on failure.
    Both inputs should NOT yet have _src / _tgt suffixes; this function
    adds them internally.
    """
    df_s = df_src.copy().add_suffix("_src")
    df_t = df_tgt.copy().add_suffix("_tgt")

    s_col = f"{src_instr_col}_src"
    t_col = f"{tgt_instr_col}_tgt"

    if s_col in df_s.columns and t_col in df_t.columns:
        df_s["cum_instr"] = df_s[s_col].fillna(0).cumsum()
        df_t["cum_instr"] = df_t[t_col].fillna(0).cumsum()
        merged = pd.merge_asof(
            df_s.sort_values("cum_instr"),
            df_t.sort_values("cum_instr"),
            on="cum_instr",
            direction="nearest",
        )
    else:
        # Fallback: truncate to same length and concat side-by-side
        n = min(len(df_s), len(df_t))
        merged = pd.concat(
            [df_s.iloc[:n].reset_index(drop=True),
             df_t.iloc[:n].reset_index(drop=True)],
            axis=1,
        )

    return merged


def prepare_bench_df(df_src, df_tgt, target_key="cpu_cycles",
                     min_instructions=100_000):
    """
    Align a source/target pair, filter invalid rows, and attach
    'target_y' and 'source_val' columns used by all worker functions.

    Returns a merged DataFrame or None if the result is unusable.
    """
    # Filter before merging
    for df, col in [(df_src, "instructions"), (df_tgt, "instructions")]:
        if col in df.columns:
            df.drop(df[df[col] <= min_instructions].index, inplace=True)

    merged = merge_by_cumulative_instructions(df_src, df_tgt)
    if merged.empty:
        return None

    tgt_col = f"{target_key}_tgt"
    src_col = f"{target_key}_src"

    if tgt_col not in merged.columns or src_col not in merged.columns:
        return None

    merged["target_y"]   = merged[tgt_col]
    merged["source_val"] = merged[src_col]
    merged = merged[(merged["target_y"] > 0) & (merged["source_val"] > 0)]

    return merged if not merged.empty else None


# =============================================================================
# 5.  METRICS
# =============================================================================

def compute_metrics(y_true, y_pred, weights=None):
    """
    Return a dict with wMAPE, MAPE, and MdAPE (all in %).
    Weights are used only for wMAPE; if None, all samples are equal.
    """
    mask = y_true > 0
    yt, yp = np.asarray(y_true)[mask], np.asarray(y_pred)[mask]
    if len(yt) == 0:
        return {"wmape": np.nan, "mape": np.nan, "mdape": np.nan}

    abs_err = np.abs(yt - yp)
    w = np.asarray(weights)[mask] if weights is not None else np.ones(len(yt))

    wmape = (np.sum(abs_err) / np.sum(yt)) * 100
    mape  = np.mean(abs_err / (yt + 1e-9)) * 100
    mdape = np.median(abs_err / (yt + 1e-9)) * 100

    return {"wmape": wmape, "mape": mape, "mdape": mdape}


# =============================================================================
# 6.  FOLD RESUME HELPER
# =============================================================================

def try_load_model(out_dir, test_bench):
    """
    If model_{test_bench}.cbm already exists in out_dir, load and return it
    so callers can skip retraining. Returns None if no cached model is found.
    """
    model_path = os.path.join(out_dir, f"model_{test_bench}.cbm")
    if not os.path.exists(model_path):
        return None
    try:
        model = CatBoostRegressor()
        model.load_model(model_path)
        print(f"  [SKIP] Loaded cached model for '{test_bench}', skipping training.")
        return model
    except Exception:
        return None


def load_fold_if_done(out_dir, test_bench, freq_ratio):
    """
    If predictions_{test_bench}.csv already exists in out_dir, recompute and
    return the metrics dict from it without retraining. Returns None otherwise.
    """
    pred_path = os.path.join(out_dir, f"predictions_{test_bench}.csv")
    if not os.path.exists(pred_path):
        return None
    try:
        df = pd.read_csv(pred_path)
        if df.empty or "target_actual" not in df.columns:
            return None
        y_true = df["target_actual"].values
        y_pred = df["target_predicted"].values
        y_src  = df["source_val"].values
        m_ml    = compute_metrics(y_true, y_pred)
        m_copy  = compute_metrics(y_true, y_src)
        m_scale = compute_metrics(y_true, y_src * freq_ratio)
        print(f"  [SKIP] '{test_bench}' already complete, loading from CSV.")
        return {
            "bench":       test_bench,
            "wmape":       m_ml["wmape"],
            "mape":        m_ml["mape"],
            "mdape":       m_ml["mdape"],
            "wmape_copy":  m_copy["wmape"],
            "mape_copy":   m_copy["mape"],
            "wmape_scale": m_scale["wmape"],
            "mape_scale":  m_scale["mape"],
        }
    except Exception:
        return None


# =============================================================================
# 7.  LOOCV DISPATCHER
#     Shared strict / non-strict leave-one-out cross-validation logic.
# =============================================================================

def build_loocv_tasks(bench_dfs, process_fold_fn, args, extra_kwargs=None):
    """
    Build a list of joblib `delayed` tasks for leave-one-(workload)-out CV.

    Parameters
    ----------
    bench_dfs       : dict[str -> pd.DataFrame]
                      Keys are e.g. 'spec_502.gcc_r_phase0'
    process_fold_fn : callable
                      Worker function; called as
                        process_fold_fn(test_bench, train_dfs, test_df, args, **extra_kwargs)
    args            : argparse.Namespace  (carries --strict_loocv)
    extra_kwargs    : dict  — forwarded verbatim to process_fold_fn

    Returns
    -------
    list of delayed() objects ready for Parallel()
    """
    extra_kwargs = extra_kwargs or {}
    valid_benches = sorted(bench_dfs.keys())

    if args.strict_loocv:
        # Group phases by base workload name  e.g. 'spec_502.gcc_r'
        groups = {}
        for b in valid_benches:
            groups.setdefault(b.split("_phase")[0], []).append(b)

        tasks = []
        for base_name, t_benches in groups.items():
            train_benches = [b for b in valid_benches if b not in t_benches]
            if not train_benches:
                continue
            train_dfs = [bench_dfs[x] for x in train_benches]
            test_df   = pd.concat([bench_dfs[x] for x in t_benches], ignore_index=True)
            tasks.append(
                delayed(process_fold_fn)(base_name, train_dfs, test_df, args, **extra_kwargs)
            )
    else:
        tasks = [
            delayed(process_fold_fn)(
                b,
                [bench_dfs[x] for x in valid_benches if x != b],
                bench_dfs[b],
                args,
                **extra_kwargs,
            )
            for b in valid_benches
        ]

    return tasks


def run_loocv(bench_dfs, process_fold_fn, args, extra_kwargs=None):
    """
    Execute LOOCV and return a list of result dicts (None results dropped).

    Uses loky backend (robust to OOM-killed workers); n_jobs taken from args.jobs.
    Falls back to sequential execution if any worker process is lost.
    """
    extra_kwargs = extra_kwargs or {}
    tasks = build_loocv_tasks(bench_dfs, process_fold_fn, args, extra_kwargs)
    if not tasks:
        return []

    print(f"  Dispatching {len(tasks)} LOOCV tasks (strict={args.strict_loocv}, "
          f"jobs={args.jobs})...")

    try:
        results = Parallel(n_jobs=args.jobs, verbose=0, backend="loky")(tasks)
        return [r for r in results if r is not None]
    except Exception as e:
        print(f"  [ERROR] Parallel execution failed ({type(e).__name__}: {e})")
        print("  Retrying sequentially...")

    # Sequential fallback: rebuild tasks as plain calls to avoid re-pickling issues
    valid_benches = sorted(bench_dfs.keys())
    results = []
    if args.strict_loocv:
        groups = {}
        for b in valid_benches:
            groups.setdefault(b.split("_phase")[0], []).append(b)
        for base_name, t_benches in groups.items():
            train_benches = [b for b in valid_benches if b not in t_benches]
            if not train_benches:
                continue
            train_dfs = [bench_dfs[x] for x in train_benches]
            test_df   = pd.concat([bench_dfs[x] for x in t_benches], ignore_index=True)
            try:
                r = process_fold_fn(base_name, train_dfs, test_df, args, **extra_kwargs)
                if r is not None:
                    results.append(r)
            except Exception as e:
                print(f"  [WARN] Sequential fold '{base_name}' failed: {e}")
    else:
        for b in valid_benches:
            train_dfs = [bench_dfs[x] for x in valid_benches if x != b]
            try:
                r = process_fold_fn(b, train_dfs, bench_dfs[b], args, **extra_kwargs)
                if r is not None:
                    results.append(r)
            except Exception as e:
                print(f"  [WARN] Sequential fold '{b}' failed: {e}")
    return results


# =============================================================================
# 7b. GENERAL-MODEL RUNNER
#     One model trained on all workloads; the complement to LOOCV. Two modes:
#       general_insample  -- train on all, evaluate on each (learnability ceiling)
#       general_temporal  -- train on the first --temporal_frac of every workload,
#                            test on the held-out tail (deployable shared model)
#     Evaluated per-workload and returned in the SAME result-dict shape as
#     run_loocv, so print_summary / save_feature_importance / grand_summary all
#     work unchanged and the numbers sit apples-to-apples next to the LOOCV row.
# =============================================================================

def _target_ratio_log(df):
    """log(clip(target_y / source_val, 0.2, 5.0)) -- identical to process_fold."""
    src_clean = df["source_val"].replace(0, np.nan).fillna(1e-9)
    ratio = df["target_y"] / src_clean
    return np.log(np.clip(ratio, 0.2, 5.0))


def run_general_model(bench_dfs, args, freq_ratio, out_dir, mode):
    """Train a single general model and evaluate it per-workload.

    Returns a list of per-workload result dicts (same schema as run_loocv);
    feature importances are attached to the first row only (they come from the
    one shared model, so averaging in save_feature_importance is a no-op)."""
    strict = getattr(args, "strict_loocv", True)
    frac = getattr(args, "temporal_frac", 0.8)
    input_counters = getattr(args, "input_counters", None)

    def base(b):
        return b.split("_phase")[0] if strict else b

    # Build the pooled training set and the per-workload evaluation groups.
    train_parts = []
    eval_groups = {}  # base workload name -> list of test-split DataFrames
    for b, df in bench_dfs.items():
        if mode == "general_temporal":
            n = int(len(df) * frac)
            if n < 1 or (len(df) - n) < 1:
                continue  # too short to split
            train_parts.append(df.iloc[:n])
            eval_groups.setdefault(base(b), []).append(df.iloc[n:])
        else:  # general_insample
            train_parts.append(df)
            eval_groups.setdefault(base(b), []).append(df)

    if not train_parts or not eval_groups:
        print(f"  [general:{mode}] nothing to train/evaluate after splitting.")
        return []

    train_full = pd.concat(train_parts, ignore_index=True)

    if getattr(args, "equal_weight", False):
        sample_weights = np.concatenate(
            [np.full(len(p), 1.0 / len(p)) for p in train_parts]
        )
    else:
        sample_weights = None

    train_full = restrict_input_counters(train_full, "_src", input_counters)
    X_train = build_features(train_full, suffix="_src", args=args)
    if X_train.empty:
        return []
    y_train_log = _target_ratio_log(train_full)

    # Pooled test used only as the early-stopping eval set (mirrors process_fold,
    # which also early-stops on its evaluation data -- keeps methodology identical
    # to LOOCV). For general_insample this is the training data itself.
    test_pooled = pd.concat(
        [d for lst in eval_groups.values() for d in lst], ignore_index=True)
    test_pooled = restrict_input_counters(test_pooled, "_src", input_counters)
    X_test_pooled = build_features(test_pooled, suffix="_src", args=args)

    common_cols = sorted(set(X_train.columns) & set(X_test_pooled.columns))
    if not common_cols:
        return []
    X_train = X_train[common_cols]
    X_test_pooled = X_test_pooled[common_cols]
    y_test_pooled_log = _target_ratio_log(test_pooled)

    print(f"  [general:{mode}] training 1 model on {len(X_train)} rows from "
          f"{len(train_parts)} workload-splits; evaluating {len(eval_groups)} workloads.")

    model = build_model(cat_feature_names(args))
    model.fit(
        X_train, y_train_log,
        eval_set=(X_test_pooled, y_test_pooled_log),
        early_stopping_rounds=200,
        sample_weight=sample_weights,
    )
    importances = dict(zip(X_train.columns.tolist(), model.get_feature_importance()))

    os.makedirs(out_dir, exist_ok=True)
    model.save_model(os.path.join(out_dir, f"model_general_{mode}.cbm"))

    results = []
    for i, (name, dfs) in enumerate(sorted(eval_groups.items())):
        tdf = restrict_input_counters(pd.concat(dfs, ignore_index=True), "_src", input_counters)
        X_t = build_features(tdf, suffix="_src", args=args)
        if X_t.empty:
            continue
        X_t = X_t.reindex(columns=common_cols, fill_value=0)

        src_cycles = tdf["source_val"].values
        y_true_cycles = tdf["target_y"].values
        pred_cycles = np.exp(model.predict(X_t)) * src_cycles

        m_ml    = compute_metrics(y_true_cycles, pred_cycles)
        m_copy  = compute_metrics(y_true_cycles, src_cycles)
        m_scale = compute_metrics(y_true_cycles, src_cycles / freq_ratio)

        row = {
            "bench":       name,
            "wmape":       m_ml["wmape"],
            "mape":        m_ml["mape"],
            "mdape":       m_ml["mdape"],
            "wmape_copy":  m_copy["wmape"],
            "mape_copy":   m_copy["mape"],
            "wmape_scale": m_scale["wmape"],
            "mape_scale":  m_scale["mape"],
        }
        if i == 0:
            row["feature_importances"] = importances
        results.append(row)

    return results


# =============================================================================
# 7.  RESULTS REPORTING
# =============================================================================

def print_summary(results, label=""):
    """Pretty-print and return a summary DataFrame from a list of result dicts."""
    if not results:
        print("  No results to report.")
        return pd.DataFrame()

    df = pd.DataFrame(results).sort_values("mape", ascending=False)
    header = f"  RESULTS{' — ' + label if label else ''}"
    print("\n" + "=" * 60)
    print(header)
    print("=" * 60)

    # Print only scalar columns
    display_cols = [c for c in df.columns if df[c].dtype != object or c == "bench"]
    print(df[display_cols].to_string(index=False))

    print("\n  --- Summary ---")
    for metric in ("wmape", "mape", "mdape"):
        if metric in df.columns:
            print(f"  Mean {metric.upper():6s}: {df[metric].mean():.2f}%")

    return df


def save_feature_importance(results, out_dir):
    """
    Average feature importances across LOOCV folds and write feature_importance.csv.

    Folds that were loaded from cache (no 'feature_importances' key) are skipped;
    the average is computed over whichever folds did train a fresh model.
    """
    imp_dicts = [
        r["feature_importances"] for r in results
        if r is not None and r.get("feature_importances") is not None
    ]
    if not imp_dicts:
        return

    avg_imp = (
        pd.DataFrame(imp_dicts)
        .fillna(0.0)
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    avg_imp.columns = ["feature", "importance"]
    avg_imp.to_csv(os.path.join(out_dir, "feature_importance.csv"), index=False)

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

def add_feature_args(parser):
    """Attach the shared feature-toggle and model flags to an argparse parser."""
    g = parser.add_argument_group("Feature toggles")
    g.add_argument("--use_mpki",           action="store_true", default=True,
                   help="Include MPKI-normalised cache/branch/TLB counters (default: on)")
    g.add_argument("--no_mpki",            dest="use_mpki", action="store_false")

    g.add_argument("--use_miss_rates",     action="store_true", default=True,
                   help="Include cache/branch/TLB miss-rate ratios (default: on)")
    g.add_argument("--no_miss_rates",      dest="use_miss_rates", action="store_false")

    g.add_argument("--use_stall_rates",    action="store_true", default=True,
                   help="Include frontend/backend stall-per-cycle rates (default: on)")
    g.add_argument("--no_stall_rates",     dest="use_stall_rates", action="store_false")

    g.add_argument("--use_bottleneck_class", action="store_true", default=True,
                   help="Include categorical bottleneck classifier (MoE gate, default: on)")
    g.add_argument("--no_bottleneck_class",  dest="use_bottleneck_class", action="store_false")

    g.add_argument("--rolling_window", type=int, default=0,
                   help="Rolling-mean window size applied to key rates. 0 = disabled.")

    g.add_argument("--jobs", type=int, default=16,
                   help="Parallel joblib workers")
    g.add_argument("--strict_loocv", action="store_true", default=True,
                   help="Group all phases of the same workload into the test set (default: on)")
    g.add_argument("--no_strict_loocv", dest="strict_loocv", action="store_false")

    g.add_argument("--exclude_features", nargs="+", default=[],
                   help="Feature names to drop from X after construction "
                        "(e.g. ixb_stall_rate sx_stall_rate).")

    g.add_argument("--input_counters", nargs="+", default=None,
                   help="Restrict raw hardware counters available for feature "
                        "construction to this list (plus instructions/cpu_cycles/"
                        "ref_cycles, always kept). Derived features whose underlying "
                        "counter is unavailable are silently skipped by build_features. "
                        "Default: use all available counters.")

    g.add_argument("--equal_weight", action="store_true", default=False,
                   help="Weight training samples so every workload contributes equally "
                        "regardless of trace length (1/len per sample). "
                        "Prevents long workloads from dominating the fit.")

    return parser


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
    * loss_function  = Quantile:alpha=0.65  (robust to outliers; slight upward bias
                       compensates for log-space shrinkage)
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
        loss_function   = "Quantile:alpha=0.65",
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

# ---------------------------------------------------------------------------
# 3a. Counter name maps — normalised base names shared across ARM and x86.
#     Each entry is (numerator_base, denominator_base).
#     Presence is checked dynamically; missing counters are silently skipped.
# ---------------------------------------------------------------------------

# Pairs used to compute miss rates  (num / denom)
MISS_RATE_PAIRS = [
    ("l1_dcache_load_misses",  "l1_dcache_loads",   "l1d_miss_rate"),
    ("l1_dcache_load_misses",  "l1d_cache",         "l1d_miss_rate_alt"),   # ARM naming
    ("l1_icache_load_misses",  "l1_icache_loads",   "l1i_miss_rate"),
    ("l1_icache_load_misses",  "l1i_cache",         "l1i_miss_rate_alt"),
    ("l2d_cache_refill",       "l2d_cache",         "l2d_miss_rate"),
    ("l3d_cache_refill",       "l3d_cache",         "l3d_miss_rate"),
    ("llc_misses",             "llc_loads",         "llc_miss_rate"),       # x86 naming
    ("branch_misses",          "branches",          "branch_miss_rate"),
    ("dtlb_load_misses",       "dtlb_loads",        "dtlb_miss_rate"),
    ("dtlb_store_misses",      "dtlb_stores",       "dtlb_store_miss_rate"),
    ("itlb_load_misses",       "itlb_loads",        "itlb_miss_rate"),
    ("branch_load_misses",     "branch_loads",      "branch_load_miss_rate"),
]

# Counters whose per-1000-instruction rate we include for MPKI features
MPKI_COUNTERS = [
    "l1_dcache_load_misses",
    "l1_icache_load_misses",
    "l1d_cache_refill",
    "l2d_cache_refill",
    "l3d_cache_refill",
    "llc_misses",           # x86
    "branch_misses",
    "dtlb_load_misses",
    "itlb_load_misses",
    "cache_misses",
]

# Stall counters normalised per cycle
STALL_COUNTERS = [
    "stalled_cycles_backend",
    "stalled_cycles_frontend",
    # Qualcomm/ARM fine-grained stalls
    "bx_stall", "decode_stall", "dispatch_stall",
    "fx_stall", "ixa_stall", "ixb_stall", "lx_stall", "sx_stall",
]

ALWAYS_KEEP_COUNTERS = ["instructions", "cpu_cycles", "ref_cycles"]


def restrict_input_counters(df, suffix, input_counters):
    """Drop raw counter columns (with the given suffix) not in
    ALWAYS_KEEP_COUNTERS or input_counters, so build_features() silently
    skips derived features whose underlying counter was removed."""
    if not input_counters:
        return df
    keep = {f"{c}{suffix}" for c in ALWAYS_KEEP_COUNTERS} | \
           {f"{c}{suffix}" for c in input_counters}
    drop_cols = [c for c in df.columns if c.endswith(suffix) and c not in keep]
    return df.drop(columns=drop_cols, errors="ignore")


# Rolling window applied to these features (when --rolling_window > 0)
ROLLING_FEATURE_NAMES = [
    "ipc", "l1d_miss_rate", "l1d_miss_rate_alt",
    "branch_miss_rate", "llc_miss_rate",
    "backend_stall_rate", "frontend_stall_rate",
    "l1d_mpki", "l3d_mpki", "branch_mpki",
]


def build_features(df, suffix, args):
    """
    Build the unified physics-informed feature matrix from a DataFrame.

    Parameters
    ----------
    df     : pd.DataFrame  — columns are already suffixed (e.g. 'instructions_src')
    suffix : str           — e.g. '_src' or '_tgt' or ''
    args   : argparse.Namespace  — carries feature-toggle flags

    Returns
    -------
    X      : pd.DataFrame  — feature matrix (never contains NaN/Inf)
    """
    X = pd.DataFrame(index=df.index)

    col_instr  = f"instructions{suffix}"
    col_cycles = f"cpu_cycles{suffix}"

    if col_instr not in df.columns or col_cycles not in df.columns:
        return X  # caller handles empty

    vec_instr  = df[col_instr].replace(0, np.nan)
    vec_cycles = df[col_cycles].replace(0, np.nan)
    inst_k     = vec_instr / 1000.0  # for MPKI denominator

    # --- Always-on base rates ---
    X["ipc"] = vec_instr / vec_cycles.fillna(1e-9)
    X["cpi"] = vec_cycles / vec_instr.fillna(1e-9)

    # x86: dynamic frequency ratio (actual / reference clock)
    ref_col = f"ref_cycles{suffix}"
    if ref_col in df.columns:
        X["dynamic_freq_ratio"] = vec_cycles / df[ref_col].replace(0, np.nan).fillna(1e-9)

    # --- MPKI features ---
    if args.use_mpki:
        for base in MPKI_COUNTERS:
            col = f"{base}{suffix}"
            if col in df.columns:
                fname = base.replace("_load_misses", "").replace("_cache_refill", "") \
                            .replace("_misses", "").replace("_", "") + "_mpki"
                # use a deterministic name derived from base
                fname = f"{base}_mpki"
                X[fname] = df[col] / inst_k.fillna(1e-9)

    # --- Miss-rate features ---
    if args.use_miss_rates:
        for num_base, den_base, feat_name in MISS_RATE_PAIRS:
            num_col = f"{num_base}{suffix}"
            den_col = f"{den_base}{suffix}"
            if num_col in df.columns and den_col in df.columns:
                den = df[den_col].replace(0, np.nan)
                X[feat_name] = df[num_col] / den.fillna(1e-9)

    # --- Stall-rate features ---
    if args.use_stall_rates:
        for base in STALL_COUNTERS:
            col = f"{base}{suffix}"
            if col in df.columns:
                X[f"{base}_rate"] = df[col] / vec_cycles.fillna(1e-9)

    # --- Clean up before rolling ---
    X = X.replace([np.inf, -np.inf], 0).fillna(0)

    # --- Bottleneck classifier (categorical MoE gate) ---
    # Built from the cleaned MPKI columns so thresholds are stable
    if args.use_bottleneck_class:
        l3_mpki_col     = next((c for c in ["l3d_cache_refill_mpki", "llc_misses_mpki",
                                             "cache_misses_mpki"] if c in X.columns), None)
        branch_mpki_col = "branch_misses_mpki" if "branch_misses_mpki" in X.columns else None
        l1i_mpki_col    = "l1_icache_load_misses_mpki" if "l1_icache_load_misses_mpki" in X.columns else None

        l3_mpki     = X[l3_mpki_col]     if l3_mpki_col     else pd.Series(0, index=X.index)
        branch_mpki = X[branch_mpki_col] if branch_mpki_col else pd.Series(0, index=X.index)
        l1i_mpki    = X[l1i_mpki_col]    if l1i_mpki_col    else pd.Series(0, index=X.index)

        conditions = [
            (l3_mpki > 2.0),
            (branch_mpki > 5.0) | (l1i_mpki > 10.0),
        ]
        choices = ["Memory_Bound", "Frontend_Bound"]
        X["bottleneck_class"] = np.select(conditions, choices, default="Compute_Bound")

    # --- Exclude specific features if requested ---
    if hasattr(args, "exclude_features") and args.exclude_features:
        X = X.drop(columns=[c for c in args.exclude_features if c in X.columns])

    # --- Rolling window (applied in-place to selected rate columns) ---
    if args.rolling_window > 0:
        w = args.rolling_window
        for feat in ROLLING_FEATURE_NAMES:
            if feat in X.columns:
                X[f"{feat}_roll{w}"] = (
                    X[feat].rolling(window=w, min_periods=1).mean()
                )

    return X.fillna(0)


def cat_feature_names(args):
    """Return the list of categorical feature names active under current args."""
    excluded = getattr(args, "exclude_features", []) or []
    if args.use_bottleneck_class and "bottleneck_class" not in excluded:
        return ["bottleneck_class"]
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

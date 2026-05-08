# dvfs_policies.py
import numpy as np

# ==========================================
# 1. STATIC PINNED POLICIES
# ==========================================
def make_static(target_cfg):
    def policy(time_mat, energy_mat, proxy_signal, configs, valid_configs, trans_lat, trans_nrg, metric):
        n_chunks = len(time_mat)
        idx = configs.index(target_cfg)
        trace, cum_m = np.zeros(n_chunks), 0.0
        
        for i in range(n_chunks):
            # No transition costs, just static execution
            step_t = time_mat[i, idx]
            step_e = energy_mat[i, idx]
            cum_m += step_e * (step_t**(2 if metric == 'ED2P' else 1))
            trace[i] = cum_m
        return trace
    return policy

# ==========================================
# 2. REACTIVE 1-TIMESTEP (1 Chunk Hindsight)
# ==========================================
def make_reactive_1_step(core_type):
    def policy(time_mat, energy_mat, proxy_signal, configs, valid_configs, trans_lat, trans_nrg, metric):
        n_chunks = len(time_mat)
        m_mat = energy_mat * (time_mat**(2 if metric == 'ED2P' else 1))
        # FIX: Ensure idx_list is sorted for consistent argmin mapping
        idx_list = sorted([configs.index(c) for c in valid_configs if c.startswith(core_type)])
        if not idx_list: return np.zeros(n_chunks)
        
        curr = idx_list[-1]
        trace, cum_m = np.zeros(n_chunks), 0.0
        for i in range(n_chunks):
            if i > 0: 
                prev = curr # Track previous state to calculate real transition cost
                curr = idx_list[np.argmin(m_mat[i-1, idx_list])]
                lat = trans_lat[prev, curr] 
                nrg = trans_nrg[prev, curr]
            else:
                lat, nrg = 0, 0
            
            step_t = lat + time_mat[i, curr]
            step_e = nrg + energy_mat[i, curr]
            cum_m += step_e * (step_t**(2 if metric == 'ED2P' else 1))
            trace[i] = cum_m
        return trace
    return policy

# ==========================================
# 3. PROACTIVE 1-TIMESTEP (1 Chunk Foresight)
# ==========================================
def make_proactive_1_step(core_type):
    def policy(time_mat, energy_mat, proxy_signal, configs, valid_configs, trans_lat, trans_nrg, metric):
        n_chunks = len(time_mat)
        idx_list = sorted([configs.index(c) for c in valid_configs if c.startswith(core_type)])
        if not idx_list: return np.zeros(n_chunks)
        
        prev = idx_list[-1]
        trace, cum_m = np.zeros(n_chunks), 0.0
        for i in range(n_chunks):
            best_val, best_curr = np.inf, prev
            for c in idx_list:
                lat = trans_lat[prev, c] if i > 0 else 0
                nrg = trans_nrg[prev, c] if i > 0 else 0
                val = (nrg + energy_mat[i, c]) * ((lat + time_mat[i, c])**(2 if metric == 'ED2P' else 1))
                if val < best_val:
                    best_val = val
                    best_curr = c
            
            curr = best_curr
            lat = trans_lat[prev, curr] if i > 0 else 0
            nrg = trans_nrg[prev, curr] if i > 0 else 0
            step_t = lat + time_mat[i, curr]
            step_e = nrg + energy_mat[i, curr]
            
            cum_m += step_e * (step_t**(2 if metric == 'ED2P' else 1))
            trace[i] = cum_m
            prev = curr
        return trace
    return policy

# ==========================================
# 4. GLOBAL PROACTIVE ORACLE (Viterbi pinned)
# ==========================================
def make_global_viterbi(core_type):
    def policy(time_mat, energy_mat, proxy_signal, configs, valid_configs, trans_lat, trans_nrg, metric):
        n_chunks = len(time_mat)
        idx_list = sorted([configs.index(c) for c in valid_configs if c.startswith(core_type)])
        if not idx_list: return np.zeros(n_chunks)
        
        sub_t = time_mat[:, idx_list]
        sub_e = energy_mat[:, idx_list]
        sub_lat = trans_lat[np.ix_(idx_list, idx_list)]
        sub_nrg = trans_nrg[np.ix_(idx_list, idx_list)]
        
        dp_m = np.zeros((n_chunks, len(idx_list)))
        parent_mat = np.zeros((n_chunks, len(idx_list)), dtype=int)
        dp_m[0, :] = sub_e[0, :] * (sub_t[0, :] ** (2 if metric == 'ED2P' else 1))
        idx_arr = np.arange(len(idx_list))
        
        for i in range(1, n_chunks):
            lat_costs = sub_lat + sub_t[i, :]
            nrg_costs = sub_nrg + sub_e[i, :]
            step_metrics = nrg_costs * (lat_costs ** (2 if metric == 'ED2P' else 1))
            vals = dp_m[i-1, :][:, None] + step_metrics
            best_prev = np.argmin(vals, axis=0)
            dp_m[i, :] = vals[best_prev, idx_arr]
            parent_mat[i, :] = best_prev
                        
        best_idx = np.argmin(dp_m[-1, :])
        path = np.zeros(n_chunks, dtype=int)
        curr = best_idx
        path[-1] = curr
        for i in range(n_chunks - 1, 0, -1):
            curr = parent_mat[i, curr]
            path[i-1] = curr
            
        trace, cum_m = np.zeros(n_chunks), 0.0
        prev = path[0]
        for i in range(n_chunks):
            lat = sub_lat[prev, path[i]] if i > 0 else 0
            nrg = sub_nrg[prev, path[i]] if i > 0 else 0
            step_t = lat + sub_t[i, path[i]]
            step_e = nrg + sub_e[i, path[i]]
            cum_m += step_e * (step_t ** (2 if metric == 'ED2P' else 1))
            trace[i] = cum_m
            prev = path[i]
        return trace
    return policy

def run_performance_governor(time_mat, energy_mat, proxy_signal, configs, valid_configs, trans_lat, trans_nrg, metric):
    """Linux 'performance' CPUfreq governor: always pins to maximum frequency."""
    n_chunks = len(time_mat)
    p_idx = sorted([configs.index(c) for c in valid_configs if c.startswith('P')])
    if not p_idx: return np.zeros(n_chunks)

    curr = p_idx[-1]
    trace, cum_m = np.zeros(n_chunks), 0.0
    for i in range(n_chunks):
        prev = curr
        curr = p_idx[-1]
        lat = trans_lat[prev, curr] if i > 0 else 0
        nrg = trans_nrg[prev, curr] if i > 0 else 0
        step_t = lat + time_mat[i, curr]
        step_e = nrg + energy_mat[i, curr]
        cum_m += step_e * (step_t**(2 if metric == 'ED2P' else 1))
        trace[i] = cum_m
    return trace

def run_intel_hwp(time_mat, energy_mat, proxy_signal, configs, valid_configs, trans_lat, trans_nrg, metric):
    n_chunks = len(time_mat)
    p_idx = sorted([configs.index(c) for c in valid_configs if c.startswith('P')])
    if not p_idx: return np.zeros(n_chunks)
    
    curr = p_idx[-1]
    trace, cum_m = np.zeros(n_chunks), 0.0
    for i in range(n_chunks):
        prev = curr
        if i > 0:
            raw_proxy = proxy_signal[i-1]
            hwp_ratio = np.clip((raw_proxy - 1.0) / 2.5, 0.0, 1.0)
            hwp_target = int(hwp_ratio * (len(p_idx) - 1))
            curr = p_idx[hwp_target]
            
        lat = trans_lat[prev, curr] if i > 0 else 0
        nrg = trans_nrg[prev, curr] if i > 0 else 0
        step_t = lat + time_mat[i, curr]
        step_e = nrg + energy_mat[i, curr]
        cum_m += step_e * (step_t**(2 if metric == 'ED2P' else 1))
        trace[i] = cum_m
    return trace

def make_proactive_n_step(core_type, horizon):
    def policy(time_mat, energy_mat, proxy_signal, configs, valid_configs, trans_lat, trans_nrg, metric):
        n_chunks = len(time_mat)
        idx_list = sorted([configs.index(c) for c in valid_configs if c.startswith(core_type)])
        if not idx_list: return np.zeros(n_chunks)
        
        sub_t = time_mat[:, idx_list]
        sub_e = energy_mat[:, idx_list]
        sub_lat = trans_lat[np.ix_(idx_list, idx_list)]
        sub_nrg = trans_nrg[np.ix_(idx_list, idx_list)]
        
        trace, cum_m = np.zeros(n_chunks), 0.0
        prev_idx = len(idx_list) - 1 # Assume starting at max frequency
        
        for i in range(n_chunks):
            window_len = min(horizon, n_chunks - i)
            dp_m = np.zeros((window_len, len(idx_list)))
            parent_mat = np.zeros((window_len, len(idx_list)), dtype=int)
            
            # Use actual current state to calculate first lookahead step costs
            lat_costs_0 = sub_lat[prev_idx, :] + sub_t[i, :]
            nrg_costs_0 = sub_nrg[prev_idx, :] + sub_e[i, :]
            dp_m[0, :] = nrg_costs_0 * (lat_costs_0 ** (2 if metric == 'ED2P' else 1))
            
            idx_arr = np.arange(len(idx_list))
            for w in range(1, window_len):
                lat_costs = sub_lat + sub_t[i+w, :]
                nrg_costs = sub_nrg + sub_e[i+w, :]
                step_metrics = nrg_costs * (lat_costs ** (2 if metric == 'ED2P' else 1))
                vals = dp_m[w-1, :][:, None] + step_metrics
                best_prev = np.argmin(vals, axis=0)
                dp_m[w, :] = vals[best_prev, idx_arr]
                parent_mat[w, :] = best_prev
            
            best_final = np.argmin(dp_m[window_len-1, :])
            curr_step = best_final
            for w in range(window_len-1, 0, -1):
                curr_step = parent_mat[w, curr_step]
            
            best_action = curr_step
            lat = sub_lat[prev_idx, best_action] if i > 0 else 0
            nrg = sub_nrg[prev_idx, best_action] if i > 0 else 0
            step_t = lat + sub_t[i, best_action]
            step_e = nrg + sub_e[i, best_action]
            
            cum_m += step_e * (step_t ** (2 if metric == 'ED2P' else 1))
            trace[i] = cum_m
            prev_idx = best_action
        return trace
    return policy

# ==========================================
# 5. LINUX ONDEMAND GOVERNOR
# ==========================================
def make_ondemand(core_type, up_thresh=0.80, down_thresh=0.20):
    """
    Linux 'ondemand' CPUfreq governor (Pallipadi & Starikovskiy, OLS 2006).
    Jumps to max frequency when normalized proxy utilization exceeds up_thresh;
    drops to min frequency when below down_thresh; otherwise holds current level.
    Proxy signal is normalized to [0,1] via clip((proxy-1)/2.5).
    """
    def policy(time_mat, energy_mat, proxy_signal, configs, valid_configs, trans_lat, trans_nrg, metric):
        n_chunks = len(time_mat)
        idx_list = sorted([configs.index(c) for c in valid_configs if c.startswith(core_type)])
        if not idx_list: return np.zeros(n_chunks)

        curr = idx_list[-1]
        trace, cum_m = np.zeros(n_chunks), 0.0
        for i in range(n_chunks):
            prev = curr
            if i > 0:
                util = np.clip((proxy_signal[i - 1] - 1.0) / 2.5, 0.0, 1.0)
                if util >= up_thresh:
                    curr = idx_list[-1]
                elif util <= down_thresh:
                    curr = idx_list[0]

            lat = trans_lat[prev, curr] if i > 0 else 0
            nrg = trans_nrg[prev, curr] if i > 0 else 0
            step_t = lat + time_mat[i, curr]
            step_e = nrg + energy_mat[i, curr]
            cum_m += step_e * (step_t ** (2 if metric == 'ED2P' else 1))
            trace[i] = cum_m
        return trace
    return policy

# ==========================================
# 6. LINUX CONSERVATIVE GOVERNOR
# ==========================================
def make_conservative(core_type, up_thresh=0.80, down_thresh=0.20):
    """
    Linux 'conservative' CPUfreq governor.
    Steps frequency up or down by exactly one level per interval, avoiding the
    large transition costs of ondemand's jumps to max/min.
    Reference: Linux kernel CPUfreq documentation.
    """
    def policy(time_mat, energy_mat, proxy_signal, configs, valid_configs, trans_lat, trans_nrg, metric):
        n_chunks = len(time_mat)
        idx_list = sorted([configs.index(c) for c in valid_configs if c.startswith(core_type)])
        if not idx_list: return np.zeros(n_chunks)

        level = len(idx_list) - 1
        prev_cfg = idx_list[level]
        trace, cum_m = np.zeros(n_chunks), 0.0
        for i in range(n_chunks):
            if i > 0:
                util = np.clip((proxy_signal[i - 1] - 1.0) / 2.5, 0.0, 1.0)
                if util >= up_thresh:
                    level = min(level + 1, len(idx_list) - 1)
                elif util <= down_thresh:
                    level = max(level - 1, 0)

            curr_cfg = idx_list[level]
            lat = trans_lat[prev_cfg, curr_cfg] if i > 0 else 0
            nrg = trans_nrg[prev_cfg, curr_cfg] if i > 0 else 0
            step_t = lat + time_mat[i, curr_cfg]
            step_e = nrg + energy_mat[i, curr_cfg]
            cum_m += step_e * (step_t ** (2 if metric == 'ED2P' else 1))
            trace[i] = cum_m
            prev_cfg = curr_cfg
        return trace
    return policy

# ==========================================
# 7. LINUX SCHEDUTIL (PELT-Based)
# ==========================================
def make_schedutil_pelt(core_type, alpha=0.25, headroom=1.25):
    """
    Linux 'schedutil' governor with PELT-style EWMA utilization tracking
    (Linux kernel v4.7+; Rafał Miłecki, Viresh Kumar, Rafael Wysocki).
    Smooths the proxy utilization signal via EWMA, then selects the minimum
    frequency level satisfying: freq >= util_ewma * headroom * max_freq.
    alpha controls EWMA decay; headroom (>1.0) reserves margin above predicted load.
    """
    def policy(time_mat, energy_mat, proxy_signal, configs, valid_configs, trans_lat, trans_nrg, metric):
        n_chunks = len(time_mat)
        idx_list = sorted([configs.index(c) for c in valid_configs if c.startswith(core_type)])
        if not idx_list: return np.zeros(n_chunks)

        util_ewma = 1.0
        curr = idx_list[-1]
        trace, cum_m = np.zeros(n_chunks), 0.0
        for i in range(n_chunks):
            prev = curr
            if i > 0:
                raw_util = np.clip((proxy_signal[i - 1] - 1.0) / 2.5, 0.0, 1.0)
                util_ewma = alpha * raw_util + (1.0 - alpha) * util_ewma
                target_ratio = min(util_ewma * headroom, 1.0)
                target_level = int(np.round(target_ratio * (len(idx_list) - 1)))
                curr = idx_list[np.clip(target_level, 0, len(idx_list) - 1)]

            lat = trans_lat[prev, curr] if i > 0 else 0
            nrg = trans_nrg[prev, curr] if i > 0 else 0
            step_t = lat + time_mat[i, curr]
            step_e = nrg + energy_mat[i, curr]
            cum_m += step_e * (step_t ** (2 if metric == 'ED2P' else 1))
            trace[i] = cum_m
        return trace
    return policy

# ==========================================
# 8. EWMA PREDICTOR (Weiser et al., 1994)
# ==========================================
def make_ewma_dvfs(core_type, alpha=0.5):
    """
    EWMA-based DVFS from Weiser et al., "Scheduling for Reduced CPU Energy" (OSDI 1994).
    Predicts next-interval utilization via exponentially weighted moving average of the
    proxy signal, then runs at the minimum frequency sufficient for that prediction.
    alpha controls how quickly the predictor tracks changes (higher = more reactive).
    """
    def policy(time_mat, energy_mat, proxy_signal, configs, valid_configs, trans_lat, trans_nrg, metric):
        n_chunks = len(time_mat)
        idx_list = sorted([configs.index(c) for c in valid_configs if c.startswith(core_type)])
        if not idx_list: return np.zeros(n_chunks)

        pred = 1.0
        curr = idx_list[-1]
        trace, cum_m = np.zeros(n_chunks), 0.0
        for i in range(n_chunks):
            prev = curr
            if i > 0:
                actual = np.clip((proxy_signal[i - 1] - 1.0) / 2.5, 0.0, 1.0)
                pred = alpha * actual + (1.0 - alpha) * pred
                target_level = int(np.ceil(pred * (len(idx_list) - 1)))
                curr = idx_list[np.clip(target_level, 0, len(idx_list) - 1)]

            lat = trans_lat[prev, curr] if i > 0 else 0
            nrg = trans_nrg[prev, curr] if i > 0 else 0
            step_t = lat + time_mat[i, curr]
            step_e = nrg + energy_mat[i, curr]
            cum_m += step_e * (step_t ** (2 if metric == 'ED2P' else 1))
            trace[i] = cum_m
        return trace
    return policy

# ==========================================
# 9. UCB1 ONLINE BANDIT DVFS
# ==========================================
def make_ucb1_dvfs(core_type, c=1.0):
    """
    UCB1 multi-armed bandit DVFS (Auer et al., Machine Learning 2002).
    Treats each available frequency as a bandit arm; reward is the negative
    per-chunk EDP/ED2P cost, normalized online by the running maximum to keep
    UCB confidence bounds scaled correctly across workloads.
    Initialization: round-robins through all arms once before exploiting.
    c controls the exploration-exploitation tradeoff (larger = more exploration).
    """
    def policy(time_mat, energy_mat, proxy_signal, configs, valid_configs, trans_lat, trans_nrg, metric):
        n_chunks = len(time_mat)
        idx_list = sorted([configs.index(c) for c in valid_configs if c.startswith(core_type)])
        if not idx_list: return np.zeros(n_chunks)
        n_arms = len(idx_list)

        counts = np.zeros(n_arms)
        values = np.zeros(n_arms)   # Mean normalized reward per arm
        max_cost_seen = 1e-12       # Running normalizer to keep rewards in [-1, 0]

        curr = idx_list[-1]
        trace, cum_m = np.zeros(n_chunks), 0.0
        for i in range(n_chunks):
            prev = curr
            if i < n_arms:
                arm = i  # Round-robin initialization: try each frequency once
            else:
                ucb = values + c * np.sqrt(2.0 * np.log(i) / counts)
                arm = int(np.argmax(ucb))

            curr = idx_list[arm]
            lat = trans_lat[prev, curr] if i > 0 else 0
            nrg = trans_nrg[prev, curr] if i > 0 else 0
            step_t = lat + time_mat[i, curr]
            step_e = nrg + energy_mat[i, curr]
            step_m = step_e * (step_t ** (2 if metric == 'ED2P' else 1))

            max_cost_seen = max(max_cost_seen, step_m)
            reward = -step_m / max_cost_seen
            counts[arm] += 1
            values[arm] += (reward - values[arm]) / counts[arm]

            cum_m += step_m
            trace[i] = cum_m
        return trace
    return policy

# ==========================================
# 10. RANDOM POLICY (Lower Bound)
# ==========================================
def make_random_dvfs(core_type, seed=42):
    """
    Random frequency selection within the specified core cluster.
    Serves as a lower bound for online learning policies (UCB1, EWMA).
    A fixed seed ensures reproducibility across workloads and metric types.
    """
    def policy(time_mat, energy_mat, proxy_signal, configs, valid_configs, trans_lat, trans_nrg, metric):
        rng = np.random.default_rng(seed)
        n_chunks = len(time_mat)
        idx_list = sorted([configs.index(c) for c in valid_configs if c.startswith(core_type)])
        if not idx_list: return np.zeros(n_chunks)

        curr = idx_list[-1]
        trace, cum_m = np.zeros(n_chunks), 0.0
        for i in range(n_chunks):
            prev = curr
            curr = idx_list[rng.integers(len(idx_list))]
            lat = trans_lat[prev, curr] if i > 0 else 0
            nrg = trans_nrg[prev, curr] if i > 0 else 0
            step_t = lat + time_mat[i, curr]
            step_e = nrg + energy_mat[i, curr]
            cum_m += step_e * (step_t ** (2 if metric == 'ED2P' else 1))
            trace[i] = cum_m
        return trace
    return policy
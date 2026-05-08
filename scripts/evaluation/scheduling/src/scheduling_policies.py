# scheduling_policies.py
import numpy as np

# ADDED proxy_signal to the arguments list here:
def run_proactive_hetero_oracle(time_mat, energy_mat, proxy_signal, configs, valid_configs, trans_lat, trans_nrg, metric):
    """Viterbi Dynamic Programming for Global Minimum."""
    n_chunks, n_configs = time_mat.shape
    if n_chunks == 0: return np.zeros(n_chunks)
    
    dp_m = np.zeros((n_chunks, n_configs))
    parent_mat = np.zeros((n_chunks, n_configs), dtype=int)
    dp_m[0, :] = energy_mat[0, :] * (time_mat[0, :] ** (2 if metric == 'ED2P' else 1))
    idx_arr = np.arange(n_configs)
    
    for i in range(1, n_chunks):
        lat_costs = trans_lat + time_mat[i, :]
        nrg_costs = trans_nrg + energy_mat[i, :]
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
        lat = trans_lat[prev, path[i]] if i > 0 else 0
        nrg = trans_nrg[prev, path[i]] if i > 0 else 0
        step_t = lat + time_mat[i, path[i]]
        step_e = nrg + energy_mat[i, path[i]]
        cum_m += step_e * (step_t ** (2 if metric == 'ED2P' else 1))
        trace[i] = cum_m
        prev = path[i]
    return trace

def run_micro_eas(time_mat, energy_mat, proxy_signal, configs, valid_configs, trans_lat, trans_nrg, metric):
    n_chunks = len(time_mat)
    p_idx = [configs.index(c) for c in valid_configs if c.startswith('P')]
    e_idx = [configs.index(c) for c in valid_configs if c.startswith('E')]
    if not p_idx or not e_idx: return np.zeros(n_chunks)
    
    curr = p_idx[-1]
    e_mid = e_idx[len(e_idx)//2]
    p_max = p_idx[-1]
    mig_nrg_cost = trans_nrg[p_max, e_mid] # Get base migration energy cost
    
    trace, cum_m = np.zeros(n_chunks), 0.0
    for i in range(n_chunks):
        prev_curr = curr
        if i > 0:
            if curr in p_idx:
                e_saving = energy_mat[i-1, curr] - energy_mat[i-1, e_mid]
                if e_saving > mig_nrg_cost: curr = e_mid
            else:
                if proxy_signal[i-1] > 2.5: curr = p_max
                
        step_t = trans_lat[prev_curr, curr] + time_mat[i, curr] if i > 0 else time_mat[i, curr]
        step_e = trans_nrg[prev_curr, curr] + energy_mat[i, curr] if i > 0 else energy_mat[i, curr]
        cum_m += step_e * (step_t**(2 if metric == 'ED2P' else 1))
        trace[i] = cum_m
    return trace

def make_reactive_n_step_fixed_freq(lookback, target_freq):
    """Reactive History Lookback locked to a specific frequency."""
    def policy(time_mat, energy_mat, proxy_signal, configs, valid_configs, trans_lat, trans_nrg, metric):
        n_chunks = len(time_mat)
        # Filter configurations to ONLY those matching the target frequency
        freq_configs = [c for c in valid_configs if target_freq in c]
        if not freq_configs: return np.zeros(n_chunks)
        
        idx_list = [configs.index(c) for c in freq_configs]
        m_mat = energy_mat * (time_mat**(2 if metric == 'ED2P' else 1))
        
        trace, cum_m = np.zeros(n_chunks), 0.0
        
        # Start on P-Core of that frequency if available, else E-core
        p_idx_list = [configs.index(c) for c in freq_configs if c.startswith('P')]
        prev_idx = p_idx_list[0] if p_idx_list else idx_list[0]
        
        for i in range(n_chunks):
            if i > 0:
                start_idx = max(0, i - lookback)
                past_perf = np.sum(m_mat[start_idx:i, idx_list], axis=0)
                best_action = idx_list[np.argmin(past_perf)]
            else:
                best_action = prev_idx
                
            lat = trans_lat[prev_idx, best_action] if i > 0 else 0
            nrg = trans_nrg[prev_idx, best_action] if i > 0 else 0
            step_t = lat + time_mat[i, best_action]
            step_e = nrg + energy_mat[i, best_action]
            
            cum_m += step_e * (step_t ** (2 if metric == 'ED2P' else 1))
            trace[i] = cum_m
            prev_idx = best_action
            
        return trace
    return policy

def make_proactive_n_step_fixed_freq(horizon, target_freq):
    """Proactive MPC Lookahead locked to a specific frequency."""
    def policy(time_mat, energy_mat, proxy_signal, configs, valid_configs, trans_lat, trans_nrg, metric):
        n_chunks = len(time_mat)
        freq_configs = [c for c in valid_configs if target_freq in c]
        if not freq_configs: return np.zeros(n_chunks)
        
        idx_list = [configs.index(c) for c in freq_configs]
        sub_t = time_mat[:, idx_list]
        sub_e = energy_mat[:, idx_list]
        sub_lat = trans_lat[np.ix_(idx_list, idx_list)]
        sub_nrg = trans_nrg[np.ix_(idx_list, idx_list)]
        
        trace, cum_m = np.zeros(n_chunks), 0.0
        p_idx_list = [i for i, c in enumerate(freq_configs) if c.startswith('P')]
        prev_idx = p_idx_list[0] if p_idx_list else 0
        
        for i in range(n_chunks):
            window_len = min(horizon, n_chunks - i)
            dp_m = np.zeros((window_len, len(idx_list)))
            parent_mat = np.zeros((window_len, len(idx_list)), dtype=int)
            
            lat_costs_0 = sub_lat[prev_idx, :] + sub_t[i, :] if i > 0 else sub_t[i, :]
            nrg_costs_0 = sub_nrg[prev_idx, :] + sub_e[i, :] if i > 0 else sub_e[i, :]
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
            
            curr_step = np.argmin(dp_m[window_len-1, :])
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

def make_global_oracle_fixed_freq(target_freq):
    """Global Viterbi DP locked to a specific frequency."""
    def policy(time_mat, energy_mat, proxy_signal, configs, valid_configs, trans_lat, trans_nrg, metric):
        n_chunks = len(time_mat)
        freq_configs = [c for c in valid_configs if target_freq in c]
        if not freq_configs: return np.zeros(n_chunks)
        
        idx_list = [configs.index(c) for c in freq_configs]
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

# ==========================================
# 5. LINUX EAS (Energy Aware Scheduling)
# ==========================================
def make_eas_hetero(alpha=0.25, perf_slack=0.10):
    """
    Linux Energy Aware Scheduling (EAS).
    At each interval: among all configs where previous-chunk time <= P_max_time * (1+perf_slack),
    select the minimum-energy configuration. Falls back to P-core if no config qualifies.
    PELT-style EWMA smooths utilization for scheduling stability.
    Ref: Quentin Perret et al., "An Energy Model for the Linux Kernel" (ELC 2018).
    """
    def policy(time_mat, energy_mat, proxy_signal, configs, valid_configs, trans_lat, trans_nrg, metric):
        n_chunks = len(time_mat)
        idx_list = [configs.index(c) for c in valid_configs]
        p_idx = sorted([configs.index(c) for c in valid_configs if c.startswith('P')])
        e_idx = sorted([configs.index(c) for c in valid_configs if c.startswith('E')])
        if not p_idx or not e_idx: return np.zeros(n_chunks)

        p_max = p_idx[-1]
        util_ewma = 1.0
        curr = p_max
        trace, cum_m = np.zeros(n_chunks), 0.0

        for i in range(n_chunks):
            prev = curr
            if i > 0:
                raw_util = np.clip((proxy_signal[i - 1] - 1.0) / 2.5, 0.0, 1.0)
                util_ewma = alpha * raw_util + (1.0 - alpha) * util_ewma

                # Performance constraint: prev-chunk time <= P_max * (1 + perf_slack)
                perf_limit = time_mat[i - 1, p_max] * (1.0 + perf_slack)
                feasible = [c for c in idx_list if time_mat[i - 1, c] <= perf_limit]
                if not feasible:
                    feasible = p_idx

                # EAS objective: minimum energy among feasible configs
                curr = min(feasible, key=lambda c: energy_mat[i - 1, c])

            lat = trans_lat[prev, curr] if i > 0 else 0
            nrg = trans_nrg[prev, curr] if i > 0 else 0
            step_t = lat + time_mat[i, curr]
            step_e = nrg + energy_mat[i, curr]
            cum_m += step_e * (step_t ** (2 if metric == 'ED2P' else 1))
            trace[i] = cum_m
        return trace
    return policy

# ==========================================
# 6. THRESHOLD MIGRATION WITH HYSTERESIS
# ==========================================
def make_threshold_migration(up_thresh=0.60, down_thresh=0.30, alpha=0.25):
    """
    Hysteresis-based big.LITTLE cluster migration (ARM GTS-style).
    Maintains EWMA of proxy utilization. Migrates E->P when EWMA exceeds up_thresh;
    migrates P->E when EWMA falls below down_thresh. The hysteresis gap between
    thresholds prevents rapid oscillation at the boundary. Within each cluster,
    always uses maximum frequency (no per-cluster DVFS).
    Ref: ARM big.LITTLE Technology: The Future of Mobile (Greenhalgh 2011);
         ARM GTS (Global Task Scheduling) white paper.
    """
    def policy(time_mat, energy_mat, proxy_signal, configs, valid_configs, trans_lat, trans_nrg, metric):
        n_chunks = len(time_mat)
        p_idx = sorted([configs.index(c) for c in valid_configs if c.startswith('P')])
        e_idx = sorted([configs.index(c) for c in valid_configs if c.startswith('E')])
        if not p_idx or not e_idx: return np.zeros(n_chunks)

        p_max, e_max = p_idx[-1], e_idx[-1]
        on_p_core = True
        util_ewma = 1.0
        curr = p_max
        trace, cum_m = np.zeros(n_chunks), 0.0

        for i in range(n_chunks):
            prev = curr
            if i > 0:
                raw_util = np.clip((proxy_signal[i - 1] - 1.0) / 2.5, 0.0, 1.0)
                util_ewma = alpha * raw_util + (1.0 - alpha) * util_ewma

                if on_p_core and util_ewma < down_thresh:
                    on_p_core = False
                elif not on_p_core and util_ewma > up_thresh:
                    on_p_core = True

                curr = p_max if on_p_core else e_max

            lat = trans_lat[prev, curr] if i > 0 else 0
            nrg = trans_nrg[prev, curr] if i > 0 else 0
            step_t = lat + time_mat[i, curr]
            step_e = nrg + energy_mat[i, curr]
            cum_m += step_e * (step_t ** (2 if metric == 'ED2P' else 1))
            trace[i] = cum_m
        return trace
    return policy

# ==========================================
# 7. EAS + DVFS (Combined Modern Linux)
# ==========================================
def make_eas_with_dvfs(alpha=0.25, headroom=1.25, perf_slack=0.10):
    """
    Combined Linux EAS + schedutil (modern Linux behavior, v5.0+).
    Step 1 (EAS): checks if any E-core frequency meets performance constraint
    (prev-chunk time <= P_max * (1+perf_slack)); if yes, selects E-core cluster.
    Step 2 (schedutil): within the chosen cluster, PELT util * headroom selects
    the frequency level, matching Linux schedutil governor behavior.
    Ref: Linux kernel v5.0+ EAS + schedutil interaction documentation.
    """
    def policy(time_mat, energy_mat, proxy_signal, configs, valid_configs, trans_lat, trans_nrg, metric):
        n_chunks = len(time_mat)
        p_idx = sorted([configs.index(c) for c in valid_configs if c.startswith('P')])
        e_idx = sorted([configs.index(c) for c in valid_configs if c.startswith('E')])
        if not p_idx or not e_idx: return np.zeros(n_chunks)

        p_max = p_idx[-1]
        util_ewma = 1.0
        curr = p_max
        trace, cum_m = np.zeros(n_chunks), 0.0

        for i in range(n_chunks):
            prev = curr
            if i > 0:
                raw_util = np.clip((proxy_signal[i - 1] - 1.0) / 2.5, 0.0, 1.0)
                util_ewma = alpha * raw_util + (1.0 - alpha) * util_ewma
                target_ratio = min(util_ewma * headroom, 1.0)

                # Step 1 (EAS): cluster selection via performance constraint
                perf_limit = time_mat[i - 1, p_max] * (1.0 + perf_slack)
                e_feasible = sorted([c for c in e_idx if time_mat[i - 1, c] <= perf_limit])

                if e_feasible:
                    # Step 2 (schedutil): scale within E-core cluster
                    level = int(np.round(target_ratio * (len(e_feasible) - 1)))
                    curr = e_feasible[np.clip(level, 0, len(e_feasible) - 1)]
                else:
                    # Step 2 (schedutil): scale within P-core cluster
                    level = int(np.round(target_ratio * (len(p_idx) - 1)))
                    curr = p_idx[np.clip(level, 0, len(p_idx) - 1)]

            lat = trans_lat[prev, curr] if i > 0 else 0
            nrg = trans_nrg[prev, curr] if i > 0 else 0
            step_t = lat + time_mat[i, curr]
            step_e = nrg + energy_mat[i, curr]
            cum_m += step_e * (step_t ** (2 if metric == 'ED2P' else 1))
            trace[i] = cum_m
        return trace
    return policy

# ==========================================
# 8. INTEL THREAD DIRECTOR STYLE
# ==========================================
def make_thread_director(window=5, compute_thresh=0.60, alpha=0.25, headroom=1.25):
    """
    Intel Thread Director (ITD) classification-based heterogeneous scheduling.
    Classifies workload type via a sliding window average of the proxy signal:
    high average proxy -> compute-heavy (benefits from P-core);
    low average proxy -> memory-bound (E-core sufficient).
    The compute score is then used to select the cluster; within the chosen
    cluster, schedutil-style frequency scaling is applied via PELT EWMA.
    Ref: Intel Thread Director / Enhanced Hardware Feedback Interface (EHFI),
         Intel Alder Lake Architecture (2021).
    """
    def policy(time_mat, energy_mat, proxy_signal, configs, valid_configs, trans_lat, trans_nrg, metric):
        n_chunks = len(time_mat)
        p_idx = sorted([configs.index(c) for c in valid_configs if c.startswith('P')])
        e_idx = sorted([configs.index(c) for c in valid_configs if c.startswith('E')])
        if not p_idx or not e_idx: return np.zeros(n_chunks)

        util_ewma = 1.0
        curr = p_idx[-1]
        trace, cum_m = np.zeros(n_chunks), 0.0

        for i in range(n_chunks):
            prev = curr
            if i > 0:
                # Sliding-window classification (ITD hardware feedback signal analog)
                start = max(0, i - window)
                avg_proxy = np.mean(proxy_signal[start:i])
                compute_score = np.clip((avg_proxy - 1.0) / 2.5, 0.0, 1.0)

                # Schedutil-style frequency scaling within chosen cluster
                raw_util = np.clip((proxy_signal[i - 1] - 1.0) / 2.5, 0.0, 1.0)
                util_ewma = alpha * raw_util + (1.0 - alpha) * util_ewma
                target_ratio = min(util_ewma * headroom, 1.0)

                if compute_score >= compute_thresh:
                    level = int(np.round(target_ratio * (len(p_idx) - 1)))
                    curr = p_idx[np.clip(level, 0, len(p_idx) - 1)]
                else:
                    level = int(np.round(target_ratio * (len(e_idx) - 1)))
                    curr = e_idx[np.clip(level, 0, len(e_idx) - 1)]

            lat = trans_lat[prev, curr] if i > 0 else 0
            nrg = trans_nrg[prev, curr] if i > 0 else 0
            step_t = lat + time_mat[i, curr]
            step_e = nrg + energy_mat[i, curr]
            cum_m += step_e * (step_t ** (2 if metric == 'ED2P' else 1))
            trace[i] = cum_m
        return trace
    return policy

# ==========================================
# 9. UCB1 HETERO BANDIT (Full P+E Space)
# ==========================================
def make_ucb1_hetero(c=1.0):
    """
    UCB1 multi-armed bandit across the full heterogeneous configuration space (P+E x all freqs).
    Treats each (core_type, frequency) pair as a bandit arm. Rewards are the negative
    per-chunk EDP/ED2P cost, normalized online by the running maximum.
    Initialization: round-robins through all valid configs once before exploiting.
    Extends make_ucb1_dvfs (single-cluster) to the full P+E decision space.
    Ref: Auer et al., "Finite-time Analysis of the Multiarmed Bandit Problem" (2002).
    """
    def policy(time_mat, energy_mat, proxy_signal, configs, valid_configs, trans_lat, trans_nrg, metric):
        n_chunks = len(time_mat)
        idx_list = [configs.index(c) for c in valid_configs]
        if not idx_list: return np.zeros(n_chunks)
        n_arms = len(idx_list)

        counts = np.zeros(n_arms)
        values = np.zeros(n_arms)
        max_cost_seen = 1e-12

        p_positions = [i for i, c in enumerate(valid_configs) if c.startswith('P')]
        start_arm = p_positions[-1] if p_positions else 0
        curr = idx_list[start_arm]

        trace, cum_m = np.zeros(n_chunks), 0.0
        for i in range(n_chunks):
            prev = curr
            if i < n_arms:
                arm = i
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
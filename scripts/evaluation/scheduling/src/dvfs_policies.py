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

def run_reactive_p_dvfs(time_mat, energy_mat, proxy_signal, configs, valid_configs, trans_lat, trans_nrg, metric):
    n_chunks = len(time_mat)
    m_mat = energy_mat * (time_mat**(2 if metric == 'ED2P' else 1))
    p_idx = sorted([configs.index(c) for c in valid_configs if c.startswith('P')])
    if not p_idx: return np.zeros(n_chunks)
    
    curr = p_idx[-1]
    trace, cum_m = np.zeros(n_chunks), 0.0
    for i in range(n_chunks):
        if i > 0: 
            prev = curr
            curr = p_idx[np.argmin(m_mat[i-1, p_idx])]
            lat, nrg = trans_lat[prev, curr], trans_nrg[prev, curr]
        else:
            lat, nrg = 0, 0
        step_t = lat + time_mat[i, curr]
        step_e = nrg + energy_mat[i, curr]
        cum_m += step_e * (step_t**(2 if metric == 'ED2P' else 1))
        trace[i] = cum_m
    return trace

def run_linux_schedutil(time_mat, energy_mat, proxy_signal, configs, valid_configs, trans_lat, trans_nrg, metric):
    n_chunks = len(time_mat)
    p_idx = sorted([configs.index(c) for c in valid_configs if c.startswith('P')])
    if not p_idx: return np.zeros(n_chunks)
    
    curr = p_idx[-1]
    trace, cum_m = np.zeros(n_chunks), 0.0
    for i in range(n_chunks):
        prev = curr
        curr = p_idx[-1] # Schedutil pegs to max
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
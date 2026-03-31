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

# Add this to scheduling_policies.py

def make_reactive_n_step_hetero(lookback):
    """N-Step History Lookback (Reactive Low-Pass Filter)"""
    def policy(time_mat, energy_mat, proxy_signal, configs, valid_configs, trans_lat, trans_nrg, metric):
        n_chunks = len(time_mat)
        if not valid_configs: return np.zeros(n_chunks)
        
        idx_list = [configs.index(c) for c in valid_configs]
        m_mat = energy_mat * (time_mat**(2 if metric == 'ED2P' else 1))
        
        trace, cum_m = np.zeros(n_chunks), 0.0
        
        # Assume thread starts on the Max P-Core if available
        p_idx_list = [i for i, c in enumerate(valid_configs) if c.startswith('P')]
        prev_idx = p_idx_list[-1] if p_idx_list else 0
        
        for i in range(n_chunks):
            if i > 0:
                # Look backwards up to 'lookback' chunks
                start_idx = max(0, i - lookback)
                # Find which configuration would have been best over that past window
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
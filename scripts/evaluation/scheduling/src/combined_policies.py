# combined_policies.py
import numpy as np

# ==========================================
# 1. REACTIVE N-STEP COMBINED (History Lookback)
# ==========================================
def make_reactive_n_step_combined(lookback):
    def policy(time_mat, energy_mat, proxy_signal, configs, valid_configs, trans_lat, trans_nrg, metric):
        n_chunks = len(time_mat)
        if not valid_configs: return np.zeros(n_chunks)
        
        # We allow transitions across the ENTIRE grid (all cores, all freqs)
        idx_list = [configs.index(c) for c in valid_configs]
        m_mat = energy_mat * (time_mat**(2 if metric == 'ED2P' else 1))
        
        trace, cum_m = np.zeros(n_chunks), 0.0
        
        # Start on Max P-Core if available
        p_idx_list = [configs.index(c) for c in valid_configs if c.startswith('P')]
        prev_idx = p_idx_list[-1] if p_idx_list else idx_list[0]
        
        for i in range(n_chunks):
            if i > 0:
                # Look backwards up to 'lookback' chunks
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

# ==========================================
# 2. PROACTIVE N-STEP COMBINED (MPC Lookahead)
# ==========================================
def make_proactive_n_step_combined(horizon):
    def policy(time_mat, energy_mat, proxy_signal, configs, valid_configs, trans_lat, trans_nrg, metric):
        n_chunks = len(time_mat)
        if not valid_configs: return np.zeros(n_chunks)
        idx_list = [configs.index(c) for c in valid_configs]
        
        sub_t = time_mat[:, idx_list]
        sub_e = energy_mat[:, idx_list]
        sub_lat = trans_lat[np.ix_(idx_list, idx_list)]
        sub_nrg = trans_nrg[np.ix_(idx_list, idx_list)]
        
        trace, cum_m = np.zeros(n_chunks), 0.0
        p_idx_list = [i for i, c in enumerate(valid_configs) if c.startswith('P')]
        prev_idx = p_idx_list[-1] if p_idx_list else 0
        
        for i in range(n_chunks):
            window_len = min(horizon, n_chunks - i)
            dp_m = np.zeros((window_len, len(idx_list)))
            parent_mat = np.zeros((window_len, len(idx_list)), dtype=int)
            
            # Step 0 evaluates transition from ACTUAL current state
            lat_costs_0 = sub_lat[prev_idx, :] + sub_t[i, :] if i > 0 else sub_t[i, :]
            nrg_costs_0 = sub_nrg[prev_idx, :] + sub_e[i, :] if i > 0 else sub_e[i, :]
            dp_m[0, :] = nrg_costs_0 * (lat_costs_0 ** (2 if metric == 'ED2P' else 1))
            
            idx_arr = np.arange(len(idx_list))
            
            # Remaining steps in window
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

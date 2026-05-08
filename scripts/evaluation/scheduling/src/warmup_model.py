import numpy as np


def apply_warmup_penalty(time_mat, policy_path, configs,
                         A_PtoE=0.0, tau_PtoE=1.0,
                         A_EtoP=0.0, tau_EtoP=1.0, K=50):
    """
    Return a copy of time_mat with per-chunk time inflation for K chunks after
    each cross-cluster migration in policy_path.

    time_mat:    (n_chunks, n_configs) array of per-chunk execution times
    policy_path: (n_chunks,) int array of config indices chosen by a policy
    configs:     list of config name strings (e.g. 'P_4.0GHz', 'E_2.0GHz')
    A, tau:      exponential decay parameters from warmup_params.csv;
                 A=0.0 disables the penalty (results match no-penalty baseline)
    K:           number of post-migration chunks to penalize

    slowdown model: time[i+k] *= 1 + A * exp(-k / tau)  for k in [0, K)
    """
    penalized = time_mat.copy()
    for i in range(1, len(policy_path)):
        prev_core = configs[policy_path[i - 1]][0]   # 'P' or 'E'
        curr_core = configs[policy_path[i]][0]
        if prev_core == curr_core:
            continue
        A   = A_PtoE   if prev_core == 'P' else A_EtoP
        tau = tau_PtoE if prev_core == 'P' else tau_EtoP
        for k in range(min(K, len(policy_path) - i)):
            penalized[i + k, policy_path[i + k]] *= (1.0 + A * np.exp(-k / tau))
    return penalized

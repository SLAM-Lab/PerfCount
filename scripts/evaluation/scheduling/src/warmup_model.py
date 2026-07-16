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
    path = np.asarray(policy_path)
    n = len(path)
    if n < 2 or (A_PtoE == 0.0 and A_EtoP == 0.0):
        return penalized

    # Core letter ('P'/'E') of the config chosen at each chunk.
    core = np.array([configs[a][0] for a in path])
    migrated = np.flatnonzero(core[1:] != core[:-1]) + 1   # chunk indices of a migration
    if migrated.size == 0:
        return penalized

    # Per-migration amplitude/decay, selected by the direction left behind.
    from_p = core[migrated - 1] == 'P'
    amps = np.where(from_p, A_PtoE, A_EtoP)
    taus = np.where(from_p, tau_PtoE, tau_EtoP)

    # Multiplicative factor per (migration, offset k), scattered onto the chunks the
    # policy actually occupies. Overlapping warmups compound, matching the original
    # loop: two migrations K chunks apart penalize the shared chunks twice.
    rows = np.arange(n)
    factor = np.ones(n)
    ks = np.arange(K)
    for m_i, (mi, A, tau) in enumerate(zip(migrated, amps, taus)):
        span = min(K, n - mi)
        factor[mi:mi + span] *= (1.0 + A * np.exp(-ks[:span] / tau))
    penalized[rows, path] = time_mat[rows, path] * factor
    return penalized

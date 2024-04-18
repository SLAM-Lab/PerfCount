import pandas as pd

from joblib import Parallel, delayed

def get_selected_ipc(series):
    sel = series.loc['select']
    ipc = series.loc[sel]
    return ipc

def get_per_sample_oracle_prefetcher(aligned_df):
    prefetchers = aligned_df.columns.get_level_values(0).unique().values
    best_pref = prefetchers[aligned_df.swaplevel(axis=1)['IPC'].apply(pd.Series.argmax, axis=1).values]
    return best_pref

def get_per_sample_oracle_ipc(aligned_df, best_pref, names=['sample_best', 'sample_reactive']):
    sel_ipc_df = aligned_df.swaplevel(axis=1)['IPC']
    sel_ipc_df['select'] = best_pref
    best_ipc = sel_ipc_df.apply(get_selected_ipc, axis=1)
    if sel_ipc_df.index.nlevels > 1:
        sel_ipc_df['select'] = sel_ipc_df['select'].unstack(0).shift().fillna('bopOnly').stack().swaplevel().sort_index().loc[sel_ipc_df.index].values
    else:
        sel_ipc_df['select'] = sel_ipc_df['select'].shift().fillna('bopOnly').values
    reactive_best_ipc = sel_ipc_df.apply(get_selected_ipc, axis=1)
    return pd.concat([best_ipc, reactive_best_ipc], axis=1, keys=names)


def generate_mcts_phases(data_set_pdf, phase_file_name,
                            start_rollouts=300, step_rollouts=30, min_rollouts=15,
                            max_fsize=50, max_k=10,
                            append=False,
                        ):
    from utils.monte_carlo_tree_search import MCTS
    from utils.use_mcts import PhaseHyperparam

    all_data = data_set_pdf
    oracle = get_per_sample_oracle_prefetcher(all_data)
    oracle_ipc = get_per_sample_oracle_ipc(all_data, oracle)
    best_mean_ipc = oracle_ipc['sample_best'].groupby(level=0).mean()
    cluster_data = all_data
    cluster_data = cluster_data.drop([feat for feat in cluster_data.columns.get_level_values(1).unique() if 'pref' in feat], axis=1, level=1)

    def train_mcts(bm, data):
        if data.shape[0] < 5:
            return bm, None, None
        best_bm_ipc = best_mean_ipc.loc[bm]
        tree = MCTS(reward_args=(best_bm_ipc, data.droplevel(0).dropna(how='all', axis=1)))
        hyperparam = PhaseHyperparam(('IPC',), 1, None, None, None, False, tuple(data.columns.get_level_values(1).unique().values), max_fsize, max_k)
        # hyperparam.set_max_fsize(max_fsize)
        # hyperparam.set_max_k(min(max_k, data.shape[0] - 1))
        # print('Min K: ', min(max_k, data.shape[0] - 1), 'BM: ', bm)
        assert hyperparam.all_features[0] == 'IPC'
        rollouts = start_rollouts
        while not hyperparam.terminal:
            for _ in range(rollouts):
                tree.do_rollout(hyperparam)
            hyperparam = tree.choose(hyperparam)
            rollouts -= step_rollouts
            rollouts = max(min_rollouts, rollouts)
        return bm, hyperparam, tree

    mcts_results = Parallel(n_jobs=-1)(delayed(train_mcts)(bm, data) for bm, data in cluster_data.groupby(level=0))

    bm_phase_hyperparams = {}
    with open(phase_file_name, 'a') as f:
        if not append:
            f.write('benchmark,features,clustering,fsize,k\n')
        else:
            print('Appending data')
        for bm, hyperparam, _ in mcts_results:
            if hyperparam:
                f.write(bm+','+ ':'.join(hyperparam.features)+','+hyperparam.clustering+','+str(hyperparam.fsize)+','+str(hyperparam.k)+'\n')
                print('\n\tbm, hyperparam.features')
                print(bm, hyperparam.features)
                bm_phase_hyperparams[bm] = hyperparam
    return bm_phase_hyperparams


def get_inputs_and_outputs(data_set_pdf, phase_file):
    from utils.models import cluster_and_predict # Avoid circular reference
    phase_params = pd.read_csv(phase_file, index_col=0)
    if data_set_pdf.index.nlevels == 3:
        # Assumes Level 0 is the suitename, which is irrelevant to this function but some function calls use it
        data_set_pdf = data_set_pdf.droplevel(0)

    def _get_data_set(bm):
        bm_phase_params = phase_params.loc[bm]
        features = bm_phase_params['features'].split(':')
        bm_data = data_set_pdf.loc[bm].swaplevel(axis=1)[features].swaplevel(axis=1).dropna(how='all', axis=1)
        clustering = bm_phase_params['clustering']
        fsize = bm_phase_params['fsize']
        k = bm_phase_params['k']
        bm_phase_ref = cluster_and_predict(bm_data, clustering, fsize, k)
        phase_ground_truth = bm_phase_ref
        for pref in phase_ground_truth.index.get_level_values(0).unique():
            phase_ground_truth.loc[pref] = bm_phase_ref.loc['bopOnly'].values
        sample_ground_truth = pd.DataFrame(
            {col : get_per_sample_oracle_prefetcher(bm_data) for col in bm_phase_ref.index.get_level_values(0).unique()},
            index=bm_data.index
        ).stack().swaplevel().sort_index()
        input_data = data_set_pdf.loc[bm].stack(0).swaplevel().sort_index()

        return bm, input_data, sample_ground_truth, phase_ground_truth
    
    inputs_outputs = Parallel(n_jobs=len(data_set_pdf.index.get_level_values(0).unique()))(delayed(_get_data_set)(bm) for bm in data_set_pdf.index.get_level_values(0).unique())
    
    input_data = {}
    phase_ground_truth = {}
    sample_ground_truth = {}

    for bm, bm_input_data, bm_sample_ground_truth, bm_phase_ground_truth in inputs_outputs:
        input_data[bm] = bm_input_data
        sample_ground_truth[bm] = bm_sample_ground_truth
        phase_ground_truth[bm] = bm_phase_ground_truth

    input_data = pd.concat(input_data)
    phase_ground_truth = pd.concat(phase_ground_truth)
    sample_ground_truth = pd.concat(sample_ground_truth)

    return input_data, phase_ground_truth, sample_ground_truth
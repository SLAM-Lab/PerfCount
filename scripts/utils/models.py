### Data processing and ML models utils
import os

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.mixture import GaussianMixture
from scipy.signal import medfilt

from utils.traces import get_processed_counters, get_prefetcher_name_to_nibble
from utils.preprocess import get_selected_ipc


def get_k_df(base_df, processed, max_k=16):
    '''
    Trains multiple kmeans instances with a k value up to 'max_k'. The 'base_df' input expects
    a pd.DataFrame with raw counter values and the 'processed' input is a boolean that states
    whether the raw values should be processed with the function get_processed_counters. It returns
    a pd.DataFrame where each row is a value of k and the columns include an inertia value and the
    max depth of a decision tree that is trained to predict the values of k.
    '''
    pdf = base_df
    if processed:
        pdf = get_processed_counters(base_df)
    # print(pdf.describe())
    scaler = StandardScaler().fit(pdf)
    scaled_df = pd.DataFrame(scaler.transform(pdf), index=pdf.index, columns=pdf.columns)
    kmeans_df = []
    for k in range(2, max_k):
        kmeans = KMeans(k)
        kmeans.fit(scaled_df)
        pdf['kmeans'] = kmeans.predict(scaled_df)
        tree = DecisionTreeClassifier()
        tree.fit(X=pdf.drop('kmeans', axis=1), y=pdf['kmeans'])
        kmeans_df.append({'k' : k, 'inertia' : kmeans.inertia_, 'max_depth' : tree.tree_.max_depth})
    kmeans_df = pd.DataFrame(kmeans_df)
    kmeans_df['inertia_decrease'] = ((kmeans_df['inertia'].shift() - kmeans_df['inertia']) / kmeans_df['inertia'].shift()).fillna(1)
    return kmeans_df

def rename_prefetchers(data_df):
    return data_df.rename(
        columns={
            'disableAll' : '0.disAll',
            'nlpOnly' : '1.NLP', 'spatialOnly' : '1.Sp', 'bopOnly' : '1.BOP',
            'disableNLPandSpatial' : '2.BOP-SBOP',
            'disableNLP' : '3.disNLP', 'disableSpatial' : '3.disSp',
            'disableNone' : '4.disNone'
        }
    )

def print_pmu_all_rates_pref_table(per_rate_big_df, baseline, processed, k):
    all_rates_df = pd.concat(per_rate_big_df.values(), keys=per_rate_big_df.keys(), axis=1).stack(0).swaplevel().swaplevel(0,1)
    base_df = all_rates_df[baseline].sort_index().dropna(how='all')
    if baseline == 'disableAll':
        base_df = base_df[[col for col in base_df.columns if 'pref' not in col]]
    pdf = base_df
    if processed:
        pdf = get_processed_counters(base_df)
    scaler = StandardScaler().fit(pdf)
    scaled_df = pd.DataFrame(scaler.transform(pdf), index=pdf.index, columns=pdf.columns)
    kmeans_df = []
    kmeans = KMeans(k)
    kmeans.fit(scaled_df)
    base_df['kmeans'] = kmeans.predict(scaled_df)
    print('\n\t\tCluster Centers\n')
    print(pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=scaled_df.columns))
    ipc_per_k = {}
    count_per_k = {}
    for prefetcher in all_rates_df.columns.get_level_values(0).unique():
        df = all_rates_df[prefetcher].dropna(how='all')
        if baseline == 'disableAll':
            df = df[[col for col in df.columns if 'pref' not in col]]
        pdf = df
        if processed:
            pdf = get_processed_counters(df)
        scaled_df = pd.DataFrame(scaler.transform(pdf), index=pdf.index, columns=pdf.columns)
        pdf['label'] = kmeans.predict(scaled_df)
        ipc_per_k[prefetcher] = pdf.groupby(['label'])['IPC'].mean()
        # print(pdf.groupby('label')['IPC'].mean())
        count_per_k[prefetcher] = ((pdf.groupby('label').count() / pdf.shape[0]).iloc[:,0]).to_dict()
    pmu_table = pd.DataFrame(ipc_per_k)
    print('\n\t\tSample Distribution\n')
    print(pd.DataFrame(count_per_k).fillna(0).sort_index())
    print('\n\t Normalized IPC table\n')
    if baseline != 'disableAll':
        pmu_table = pmu_table.drop('disableAll', axis=1)
    print(pmu_table.div(pmu_table[baseline],axis=0).sort_index())

def print_pmu_pref_table(big_df, rate, baseline, processed, k):
    base_df = big_df[rate][baseline].dropna(how='all')
    if baseline == 'disableAll' or baseline == '0.disAll':
        base_df = base_df[[col for col in base_df.columns if 'pref' not in col]]
    pdf = base_df
    if processed:
        pdf = get_processed_counters(base_df)
    scaler = StandardScaler().fit(pdf)
    scaled_df = pd.DataFrame(scaler.transform(pdf), index=pdf.index, columns=pdf.columns)
    kmeans_df = []
    kmeans = KMeans(k, random_state=42)
    kmeans.fit(scaled_df)
    base_df['kmeans'] = kmeans.predict(scaled_df)
    cluster_center_df = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=scaled_df.columns)
    # print(pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=scaled_df.columns))
    ipc_per_k = {}
    count_per_k = {}
    for prefetcher in big_df[rate].columns.get_level_values(0).unique():
        df = big_df[rate][prefetcher].dropna(how='all')
        if baseline == 'disableAll':
            df = df[[col for col in df.columns if 'pref' not in col]]
        pdf = df
        if processed:
            pdf = get_processed_counters(df)
        scaled_df = pd.DataFrame(scaler.transform(pdf), index=pdf.index, columns=pdf.columns)
        pdf['label'] = kmeans.predict(scaled_df)
        ipc_per_k[prefetcher] = pdf.groupby(['label'])['IPC'].mean()
        count_per_k[prefetcher] = ((pdf.groupby('label').count() / pdf.shape[0]).iloc[:,0]).to_dict()
    pmu_table = pd.DataFrame(ipc_per_k)
    if baseline != 'disableAll':
        pmu_table = pmu_table.drop('disableAll', axis=1)
    norm_ipc_df = rename_prefetchers(pmu_table.div(pmu_table[baseline],axis=0).sort_index()).sort_index(axis=1)
    best_table = {}
    for idx in norm_ipc_df.index:
        best_table[idx] = norm_ipc_df.loc[idx].sort_values(ascending=False).index.values
    best_order_df = pd.DataFrame(best_table, index=['best'] + ['best_'+str(i+2) for i in range(len(norm_ipc_df.columns) - 1)]).T
    print('\n\t\tCluster Centers and their best prefetchers\n')
    print(pd.concat([cluster_center_df, best_order_df.iloc[:,:2]], axis=1, keys=['Cluster centers', 'Prefetcher order']))
    print('\n\t\tSample Distribution\n')
    print(rename_prefetchers(pd.DataFrame(count_per_k).fillna(0).sort_index()).sort_index(axis=1))
    print('\n\t Normalized IPC table\n')
    print(norm_ipc_df)
    
    
    
def get_pmu_pref_tables(pref_df, baseline, processed, k, filt=0, clustering='kmeans'):
    def preprocess_df(df, scaler=None):
        if baseline == 'disableAll':
            df = df[[col for col in df.columns if 'pref' not in col]]
        pdf = df
        if processed:
            pdf = get_processed_counters(df)
        if filt > 0:
            filtpdf = pd.concat({
                bm : pd.DataFrame(
                    {col : medfilt(pdf.loc[bm, col], filt) for col in pdf.columns},
                    index=pdf.loc[bm].index) for bm in pdf.index.get_level_values(0).unique()
                })
        else:
            filtpdf = pdf
        if not scaler:
            scaler = StandardScaler().fit(filtpdf)
        scaled_df = pd.DataFrame(scaler.transform(filtpdf), index=pdf.index, columns=pdf.columns)
        return scaled_df, scaler
    
    base_df = pref_df[baseline].dropna(how='all')
    scaled_df, scaler = preprocess_df(base_df)
    if clustering == 'gmm':
        cluster = GaussianMixture(k, random_state=42)
    else:
        cluster = KMeans(k, random_state=42, n_init='auto')
    cluster.fit(scaled_df)
    if clustering == 'gmm':
        cluster_df = pd.DataFrame(scaler.inverse_transform(cluster.means_), columns=scaled_df.columns)
    else:
        cluster_df = pd.DataFrame(scaler.inverse_transform(cluster.cluster_centers_), columns=scaled_df.columns)
    ipc_per_k = {}
    count_per_k = {}
    for prefetcher in pref_df.columns.get_level_values(0).unique():
        df = pref_df[prefetcher].dropna(how='all')
        scaled_df, _ = preprocess_df(df, scaler)
        pdf = df
        if 'IPC' not in pdf.columns:
            pdf = get_processed_counters(df)
        pdf['label'] = cluster.predict(scaled_df)
        ipc_per_k[prefetcher] = pdf.groupby(['label'])['IPC'].mean()
        count_per_k[prefetcher] = ((pdf.groupby('label').count() / pdf.shape[0]).iloc[:,0]).to_dict()
    pmu_table = pd.DataFrame(ipc_per_k)
    if baseline != 'disableAll':
        pmu_table = pmu_table.drop('disableAll', axis=1)
    norm_df = pmu_table.div(pmu_table[baseline],axis=0).sort_index()
    best_table = {}
    for idx in norm_df.index:
        best_table[idx] = norm_df.loc[idx].sort_values(ascending=False).index.values
    center_pref_df = pd.concat([cluster_df, pd.DataFrame(best_table, index=['best'] + ['best_'+str(i+2) for i in range(len(norm_df.columns) - 1)]).T], axis=1, keys=['Cluster centers', 'Prefetcher order'])
    return norm_df, center_pref_df, scaler, cluster

def cluster_and_predict(data, clustering, fsize, k):

    filtpdf = data
    if fsize > 1:
        filtpdf = pd.DataFrame({col : medfilt(filtpdf[col], fsize) for col in filtpdf.columns}, index=filtpdf.index)
    norm_ipc, pref_centers, scaler_da, model = get_pmu_pref_tables(filtpdf, 'disableAll', False, k, filt=0, clustering=clustering)
    filtpdf = filtpdf.stack(0).swaplevel().sort_index()
    train_df = filtpdf[pref_centers['Cluster centers'].columns].copy()
    scaled_df = pd.DataFrame(scaler_da.transform(train_df), columns=train_df.columns, index=train_df.index)

    phase_convert = pref_centers['Prefetcher order']['best'].to_dict()

    inp = scaled_df
    sim = pd.Series(model.predict(inp), index=inp.index).replace(phase_convert)
    return sim

def cluster_and_simipc(data, clustering, fsize, k, base_pref):

    filtpdf = data
    if fsize > 1:
        filtpdf = pd.DataFrame({col : medfilt(filtpdf[col], fsize) for col in filtpdf.columns}, index=filtpdf.index)
    norm_ipc, pref_centers, scaler_da, model = get_pmu_pref_tables(filtpdf, 'disableAll', False, k, filt=0, clustering=clustering)
    filtpdf = filtpdf.stack(0).swaplevel().sort_index()
    train_df = filtpdf[pref_centers['Cluster centers'].columns].copy()
    scaled_df = pd.DataFrame(scaler_da.transform(train_df), columns=train_df.columns, index=train_df.index)

    phase_convert = pref_centers['Prefetcher order']['best'].to_dict()

    inp = scaled_df.loc[base_pref]
    sim = pd.Series(model.predict(inp), index=inp.index).replace(phase_convert)
    dif = sim != base_pref
    while dif.any():
        ch0 = inp.index[dif][0]
        p_new = sim.loc[ch0]
        # print(ch0, p_new)
        inp = scaled_df.loc[p_new].loc[ch0+1:]
        if inp.empty:
            break
        sim.loc[inp.index] = pd.Series(model.predict(inp)).replace(phase_convert).values
        dif = sim.loc[inp.index] != p_new
    sel_ipc_df = data.swaplevel(axis=1)['IPC']
    sel_ipc_df['select'] = sim
    ipc = sel_ipc_df.apply(get_selected_ipc, axis=1)
    sel_ipc_df['select'] = sel_ipc_df['select'].shift().fillna('bopOnly').loc[sel_ipc_df.index].values
    ipc_react = sel_ipc_df.apply(get_selected_ipc, axis=1)
    return ipc.mean(), ipc_react.mean()
    

def create_tree_c_function(X, tree, func_name, file, max_depth=10, mpro=False):
    pref_nibbles = get_prefetcher_name_to_nibble()
    with open(file, 'a') as f:
        f.write('\ntree_and_depth get_%s() {\n'%(func_name))
        f.write('\tstatic tree_node prefetchers_tree[] = {\n')
        children_left = tree.tree_.children_left
        children_right = tree.tree_.children_right
        feature = tree.tree_.feature
        value = tree.tree_.value
        threshold = tree.tree_.threshold
        class_names=[pref_nibbles[c] for c in tree.classes_]
        stack = [(0, 0, None)]  # start with the root node id (0) and its depth (0)
        tree_depth = 0
        f.write('\t\t// Level 1\n')
        while tree_depth < max_depth:
            # `pop` ensures each node is only visited once
            node_id, depth, leaf_class = stack.pop(0)

            if depth == max_depth:                                                                             
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
                break

            if depth != tree_depth:
                tree_depth = depth
                f.write('\t\t// Level %d\n'%(tree_depth+1))
            
            if node_id == -1: # Artificial child of a leaf
                if mpro:
                    f.write("\t\t{0, 0, 0x%s, 0x%s},\n"%(leaf_class, leaf_class))
                else:
                    f.write("\t\t{0, 0, '%s', '%s'},\n"%(leaf_class, leaf_class))
                stack.append((-1, depth+1, leaf_class))
                stack.append((-1, depth+1, leaf_class))
            else:
                # If the left and right child of a node is not the same we have a split
                # node
                is_split_node = children_left[node_id] != children_right[node_id]
                # If a split node, append left and right children and depth to `stack`
                # so we can loop through them
                if is_split_node:
                    stack.append((children_left[node_id], depth + 1, None))
                    stack.append((children_right[node_id], depth + 1, None))
                    line = '\t\t{'
                    if mpro:
                        line += "{feature}, {threshold}, 0x{trueclass}, 0x{falseclass},".format(
                            feature='IDX_' + X.columns[feature[node_id]].upper().replace('-', '_'),
                            threshold=round(threshold[node_id],2),
                            trueclass=class_names[np.argmax(value[children_left[node_id]])],
                            falseclass=class_names[np.argmax(value[children_right[node_id]])]
                        )
                    else:
                        line += "{feature}, {threshold}, '{trueclass}', '{falseclass}',".format(
                            feature='IDX_' + X.columns[feature[node_id]].upper().replace('-', '_'),
                            threshold=round(threshold[node_id],2),
                            trueclass=class_names[np.argmax(value[children_left[node_id]])],
                            falseclass=class_names[np.argmax(value[children_right[node_id]])]
                        )
                    line += '},\n'
                    f.write(line)
                else:
                    leaf_class = class_names[np.argmax(value[node_id])]
                    stack.append((-1, depth+1, leaf_class))
                    stack.append((-1, depth+1, leaf_class))
                    if mpro:
                        f.write("\t\t{0, 0, 0x%s, 0x%s},\n"%(leaf_class, leaf_class))
                    else:
                        f.write("\t\t{0, 0, '%s', '%s'},\n"%(leaf_class, leaf_class))
        f.write('\t};\n')
        f.write('\tstatic tree_and_depth stree;\n')
        f.write('\tstree.max_depth = %d;\n'%(max_depth))
        f.write('\tstree.tree = prefetchers_tree;\n')
        f.write('\treturn stree;\n')
        f.write('}\n')
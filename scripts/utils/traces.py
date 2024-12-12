import pandas as pd
import numpy as np
import os
from math import ceil
from joblib import Parallel, delayed

def pair_trace_alignment(A, B):
    '''
    Given two normalized traces, A and B, it generate two index
    arrays, alA and alB, that align the two traces based on the
    increased step size per sample
    '''
    assert np.round(np.sum(A),1) == 1
    assert np.round(np.sum(B),1) == 1
    n, m = len(A), len(B)
    alA, alB = np.zeros_like(A).astype(int), np.zeros_like(B).astype(int)
    tA, tB = A[0], B[0]
    i, j, k = 1, 1, 0
    while (i < n or j < m):
        if i == n: tA = nA = 1.01
        else: nA = tA + A[i]
        if j == m: tB = nB = 1.01
        else: nB = tB + B[j]
        if nA <= tB:
            assert i < n
            alA[i] = k
            tA += A[i]
            i += 1
        elif nB <= tA:
            assert j < m
            alB[j] = k
            tB += B[j]
            j += 1
        else:
            t = max(tA, tB)
            if (nA <= nB) and (abs(nA - t) < abs(nA - nB)):
                assert i < n, f"i out of bounds, got: {i, j, k, n, m, tA, tB, nA, nB}"
                alA[i] = k
                tA += A[i]
                i += 1
            elif (nB < nA) and (abs(nB - t) < abs(nA - nB)):
                assert j < m
                alB[j] = k
                tB += B[j]
                j += 1
            else:
                k += 1
                assert (i < n) and (j < m), f"i or j out of bounds, got: {i, j, k, n, m, tA, tB, nA, nB}"
                alA[i] = k
                alB[j] = k
                tA += A[i]
                tB += B[j]
                i += 1
                j += 1
    return alA, alB


def align_prefetchers_df(rdf):
    '''
    Algins traces based on # of instructions.
    Inputs: a dataframe with the following structure:
     * The columns have a multi-level index:
       - Level 0 is the name of the trace
       - Level 1 are hardware counters names.
     * The index is a multi-level index:
       - Level 0 is the name of the benchmark
       - Level 1 is the timestamp
    Outputs:
     (1) The aligned dataframe with the same structure as the input
     (2) A two-level dictionary with the index transformation from
        the aligned traces to the original form. The first level of
        the dictionary is the benchmark and the second is the name
        of the trace
    
    '''
    prefetchers = rdf.columns.get_level_values(0).unique()
    aligned_df, alignment_idxs = {}, {}
    # for bm in rdf.index.get_level_values(0).unique():
    def _alignment(bm):
        aligned_traces = [rdf.loc[bm][prefetchers[0]].dropna().drop(['total_instructions', 'IPC'], axis=1, errors='ignore')]
        idx_transforms = []
        # alignment_idxs[bm] = {}
        alignment_idxs = {}
        for pi in range(1, len(prefetchers)):
            pref1 = prefetchers[pi-1]
            pref2 = prefetchers[pi]
            df1 = aligned_traces[-1].astype(int)
            df2 = rdf.loc[bm][pref2].dropna().drop(['total_instructions', 'IPC'], axis=1, errors='ignore').astype(int)

            A = (df1['instructions'] / df1['instructions'].sum()).values
            B = (df2['instructions'] / df2['instructions'].sum()).values

            idxA, idxB = pair_trace_alignment(A, B)

            if len(idx_transforms) == 0:
                idx_transforms.append(pd.Series(idxA))
            else:
                idx_replace = pd.DataFrame(idxA, index = df1.index).to_dict()[0]

            for ti, trace in enumerate(aligned_traces):
                A = trace.groupby(idxA)[df1.columns].sum().reset_index(drop=True)
                aligned_traces[ti] = A
                if len(idx_transforms) > 1:
                    idx_transforms[ti] = idx_transforms[ti].replace(idx_replace)
            B = df2.groupby(idxB)[df2.columns].sum().reset_index(drop=True)

            aligned_traces.append(B)
            idx_transforms.append(pd.Series(idxB))

        for ti in range(len(aligned_traces)):
            aligned_traces[ti]['IPC'] = aligned_traces[ti]['instructions'] / aligned_traces[ti]['cpu-cycles']
            # alignment_idxs[bm][prefetchers[ti]] = idx_transforms[ti]
            alignment_idxs[prefetchers[ti]] = idx_transforms[ti]
        # aligned_df[bm] = pd.concat(aligned_traces, axis=1, keys=prefetchers)
        aligned_df = pd.concat(aligned_traces, axis=1, keys=prefetchers)

        return bm, aligned_df, alignment_idxs

    aligned_data = Parallel(n_jobs=50)(delayed(_alignment)(bm) for bm in rdf.index.get_level_values(0).unique())

    for bm, bm_aligned_df, bm_alignment_idxs in aligned_data:
        aligned_df[bm] = bm_aligned_df
        alignment_idxs[bm] = bm_alignment_idxs

    return pd.concat(aligned_df), alignment_idxs

def get_df_without_per_core(file_location):
    '''
    Returns a data frame of a Linux perf log/file generated with the \"-x ,\" option.
    It assumes that the perf counters data is aggregated and not per-core.
    Input: file location string
    Output: pandas.DataFrame with timestamp as index and counter names as columns
    '''
    df = pd.read_csv(
        file_location,
        comment='#',
        header=None,
        # names=['timestamp','value','unit','counter', 'unknown1', 'unknown2','IPC','IPC2'],
    )
    df = df.rename(columns = {0 : 'timestamp', 1 :'value', 2 : 'unit', 3 : 'counter'})
    df = df.replace('<not supported>', pd.NA, regex=False)
    df = df.replace('<not counted>', pd.NA, regex=False)
    df = df.replace('<NA>', pd.NA, regex=False)
    df = df.dropna(how='all', axis=1).dropna(subset=['value']).astype({'value' : int})
    # if 'IPC' in df.columns.values:
    #     ipc = df.groupby(['timestamp'])[['IPC']].mean().stack()
    # else:
    #     print('Error: IPC column not found')
    #     return pd.DataFrame()
    # print(df)
    if 'value' in df.columns.values and 'counter' in df.columns.values:
        counters = df.groupby(['timestamp', 'counter'])['value'].mean()
    else:
        print('Error: Counter values columns not found')
        return pd.DataFrame()
    return counters.sort_index().unstack()

def get_perf_df_per_core(file_location):
    df = pd.read_csv(
        file_location,
        comment='#',
        header=None,
        names=['timestamp','coreID', 'unknown', 'value','unit','counter', 'unknown1', 'unknown2','IPC','IPC2'],
    )
    df = df.replace('<not supported>', pd.NA, regex=False)
    df = df.replace('<not counted>', pd.NA, regex=False)
    df = df.dropna(how='all', axis=1)
    return df.set_index(['timestamp', 'coreID', 'counter'])['value'].unstack('counter')

def get_trace_names_tree(folder):
    '''
    Returns a nested dictionary with a list of 'csv' files found in the 'folder' directory.
    The assumtion is that the files are structured as follows:
            <prefix>_<bm>_r<n>_<suffix>.csv
    where:
        bm = benchmark name
        n = benchmark rate
    <bm> and <n> are used as dictionary keys for the corresponding list of files
    Input: folder string
    Output: dictionary
    '''
    trace_names = {}
    for file in os.listdir(folder):
        if '.csv' not in file:
            continue
        elements = file.split('_')
        if len(elements) != 4:
            continue
        bm = elements[1]
        rate = elements[2]
        if bm not in trace_names.keys():
            trace_names[bm] = {rate : []}
        elif rate not in trace_names[bm].keys():
            trace_names[bm][rate] = []
        trace_names[bm][rate].append(file)
    return trace_names

def parse_folder(folder):
    '''
    Returns a dictionary obtained from get_trace_names_tree indexed by prefetcher.
    It assumes a directory structure rooted in the 'folder' input:
    |- folder
    |- - <prefetcher_x>
    |- - - <prefix>_<bm>_r<n>_<suffix>.csv
    '''
    prefetchers_tree = {}
    for prefetcher in os.listdir(folder):
        pref_path = os.path.join(folder, prefetcher)
        if os.path.isdir(pref_path):
            tree = get_trace_names_tree(pref_path)
            prefetchers_tree[prefetcher] = tree
    return prefetchers_tree

def create_availability_table(traces_tree):
    table = []
    for prefetcher in traces_tree.keys():
        df = ~pd.DataFrame(traces_tree[prefetcher]).isnull().sort_index().T.sort_index()
        table.append(df)
    table = pd.concat(table, axis=1, keys=list(traces_tree.keys())).fillna(False)
    return table

def create_per_rate_dfs(rates, benchmarks, all_prefetchers, traces_tree, folder, per_core=False):
    per_rate_big_df = {}
    for rate in rates:
        per_rate_big_df[rate] = []
        for bm in benchmarks:
            data_df = []
            for prefetcher in all_prefetchers:
                if bm not in traces_tree[prefetcher].keys():
                    data_df.append(pd.DataFrame())
                    continue
                if rate not in traces_tree[prefetcher][bm].keys():
                    data_df.append(pd.DataFrame())
                    continue
                location = os.path.join(folder, prefetcher, traces_tree[prefetcher][bm][rate][0])
                if not per_core:
                    df = get_df_without_per_core(location)
                else:
                    df = get_perf_df_per_core(location)
                df['IPC'] = df['instructions'] / df['cpu-cycles']
                # df['total_instructions'] = df['instructions'].cumsum()
                data_df.append(df)
            data_df = pd.concat(data_df, axis=1, keys=all_prefetchers).sort_index()
            per_rate_big_df[rate].append(data_df)
        per_rate_big_df[rate] = pd.concat(per_rate_big_df[rate], keys=benchmarks, names=['benchmark'])
    return per_rate_big_df

def get_spec_big_df(folder, per_core=False):
    traces_tree = parse_folder(folder)
    table = create_availability_table(traces_tree)
    if 'sbopOnly' in table.columns.get_level_values(0):
        table = table.drop('sbopOnly', axis=1)
    per_rate_big_df = create_per_rate_dfs(
        table.columns.get_level_values(1).unique(),
        table.index.values,
        table.columns.get_level_values(0).unique(),
        traces_tree,
        folder,
        per_core,
    )
    return per_rate_big_df


def get_processed_counters(df):
    '''
    Process PMU hardware counters into normalized metrics
    It assumes that the following column names exist in the input 'df':
        instructions, branch-misses, cache-misses, mem_access,
        l2-prefetch-refill, l2c-inst-refill, l2c-data-refill, l2-prefetch-req
    Input: Raw counters data frame
    Output: Processed data frame
    '''
    if 'mem_access' in df.columns:
        mem_access = 'mem_access'
    elif 'mem-access' in df.columns:
        mem_access = 'mem-access'
    else:
        mem_access = None

    newdf = pd.DataFrame(index=df.index)
    newdf['IPC'] = 1.0 * df['instructions'] / df['cpu-cycles']
    if 'branch-misses' in df.columns:
        newdf['br-mpki'] = 1000 * df['branch-misses'] / df['instructions']
        if 'l2c-inst-refill' in df.columns:
            newdf['inst-refill-to-br-miss-ratio'] = 1.0 * df['l2c-inst-refill'] / df['branch-misses']
    
    if 'cache-misses' in df.columns:
        newdf['cache-mpki'] = 1000 * df['cache-misses'] / df['instructions']
        if 'l2c-data-refill' in df.columns:
            newdf['data-refill-to-cache-miss-ratio'] = 1.0 * df['l2c-data-refill'] / df['cache-misses']
        if 'l2-prefetch-refill' in df.columns:
            newdf['pref-refill-to-cache-miss-ratio'] = 1.0 * df['l2-prefetch-refill'] / df['cache-misses']
        if 'l2d-cache-refill' in df.columns:
            newdf['l2-refill-to-cache-miss-ratio'] = 1.0 * df['l2d-cache-refill'] / df['cache-misses']
    
    if mem_access:
        newdf['mem-apki'] = 1000 * df[mem_access] / df['instructions']
        if 'cache-misses' in df.columns:
            newdf['cache-miss-to-mem-acc-ratio'] = 1.0 * df['cache-misses'] / df[mem_access]    
        if 'l2-prefetch-req' in df.columns:
            newdf['pref-req-to-mem-acc-ratio'] = df['l2-prefetch-req'] / df[mem_access]
    
    if 'l2-prefetch-refill' in df.columns:
        if 'l2c-inst-refill' in df.columns and 'l2c-data-refill' in df.columns:
            newdf['l2-pref-mr'] = df['l2-prefetch-refill'] / (df['l2-prefetch-refill'] + df['l2c-inst-refill'] + df['l2c-data-refill'])
        elif 'l2d-cache-refill' in df.columns:
            newdf['l2-pref-mr'] = df['l2-prefetch-refill'] / df['l2d-cache-refill']
        if 'l2-prefetch-req' in df.columns:
            newdf['pref-refill-req-rate'] = df['l2-prefetch-refill'] / df['l2-prefetch-req']
    
    if 'flush' in df.columns:
        newdf['flush-pki'] = 1000 * df['flush'] / df['instructions']
    
    if 'bus-req-read' in df.columns:
        newdf['bus-req-rdpki'] = 1000 * df['bus-req-read'] / df['instructions']
        if 'bus-req-write' in df.columns:
            newdf['bus-req-pki'] = 1000 * (df['bus-req-read'] + df['bus-req-write']) / df['instructions']
    
    if 'l2d-cache-refill' in df.columns:
        newdf['l2-rpki'] = 1000 * df['l2d-cache-refill'] / df['instructions']

    if 'stall-backend-mem' in df.columns:
        newdf['stall-rate'] = df['stall-backend-mem'] / df['cpu-cycles']
        newdf['stall-pi'] = df['stall-backend-mem'] / df['instructions']
            
    return newdf.fillna(0)


def get_suite_results_and_data(folder):
    results = {}
    data = {}
    cpus = {}
    cpu_rename_done = False
    for pref_folder in os.listdir(folder):
        data_folder = os.path.join(folder, pref_folder)
        if os.path.isdir(data_folder):
            benchmarks = [file for file in os.listdir(data_folder) if file[-4:] == '.csv']
            results_file = os.path.join(data_folder, 'result.txt')
            pref_name = pref_folder.split('.')[0]
            results[pref_name] = {}
            with open(results_file) as f:
                for line in f:
                    if 'time elapsed' in line:
                        bm_name = line.split('/')[-1].split('.')[0]
                        runtime = float(line.split()[1])
                        results[pref_name][bm_name] = runtime
                    elif 'utilized' in line:
                        if not cpu_rename_done:
                            bm_name = line.split('/')[-1].split('.')[0]
                            cpus[bm_name] = str(ceil(float(line.split()[5]))) + '_' + bm_name
                cpu_rename_done
            df = pd.concat({file[:-4] : get_df_without_per_core(os.path.join(data_folder, file)) for file in benchmarks})
            df['IPC'] = df['instructions'] / df['cpu-cycles']
            data[pref_name] = df
    
    return pd.concat(data, axis=1), pd.DataFrame(results), cpus


def get_suite_results(folder):
    results = {}
    data = {}
    cpus = {}
    cpu_rename_done = False
    for pref_folder in os.listdir(folder):
        data_folder = os.path.join(folder, pref_folder)
        if os.path.isdir(data_folder):
            benchmarks = [file for file in os.listdir(data_folder) if file[-4:] == '.csv']
            results_file = os.path.join(data_folder, 'result.txt')
            pref_name = pref_folder.split('.')[0]
            results[pref_name] = {}
            with open(results_file) as f:
                for line in f:
                    if 'time elapsed' in line:
                        bm_name = line.split('/')[-1].split('.')[0]
                        runtime = float(line.split()[1])
                        results[pref_name][bm_name] = runtime
                    elif 'utilized' in line:
                        if not cpu_rename_done:
                            bm_name = line.split('/')[-1].split('.')[0]
                            cpus[bm_name] = str(ceil(float(line.split()[5]))) + '_' + bm_name
                cpu_rename_done
    
    return pd.DataFrame(results), cpus


def get_sir17_rap_traces(folder):
    traces = {}
    for pref in os.listdir(folder):
        pref_folder = os.path.join(folder, pref)
        csv_files = [file for file in os.listdir(pref_folder) if file[-4:] == '.csv']
        pref_traces = {}
        for file in csv_files:
            df = pd.read_csv(os.path.join(pref_folder, file), skipinitialspace=True)
            pieces = file[:-4].split('_')
            bm = pieces[2]
            rate = pieces[3]
            pref_traces[(bm, rate)] = df
        traces[pref] = pd.concat(pref_traces)
    return pd.concat(traces, axis=1)


def get_ren_traces(ren_folder):
    traces = {}
    for pref in os.listdir(ren_folder):
        traces[pref] = {}
        for run_name in os.listdir(os.path.join(ren_folder, pref)):
            runid = run_name.split('.')[-1]
            run_folder = os.path.join(ren_folder, pref, run_name)
            csv_files = [file for file in os.listdir(run_folder) if file[-4:] == '.csv']
            if len(csv_files) > 0:
                traces[pref][runid] = {}
                for file in csv_files:
                    df = pd.read_csv(os.path.join(run_folder, file), skipinitialspace=True)
                    if df.empty:
                        print('Empty df at ', os.path.join(run_folder, file))
                        continue
                    if (df.dropna(axis=1) == 0).all(axis=1).all():
                        print('All Zero df at ', os.path.join(run_folder, file))
                        continue
                    bm = file.split('.')[0]
                    traces[pref][runid][bm] = df
                if len(traces[pref][runid]) > 0:
                    traces[pref][runid] = pd.concat(traces[pref][runid])
                    newidx = traces[pref][runid].index.set_names('benchmark', level=0)
                    traces[pref][runid].set_index(newidx, inplace=True)
                else:
                    traces[pref].pop(runid)
        
        if len(traces[pref]) > 0:
            traces[pref] = pd.concat(traces[pref])
            newidx = traces[pref].index.set_names('runid', level=0)
            traces[pref].set_index(newidx, inplace=True)
        else:
            traces.pop(pref)
    
    traces = pd.concat(traces)
    newidx = traces.index.set_names('prefetcher', level=0)
    traces.set_index(newidx, inplace=True)

    return traces


def clean_aligned_ipc(aligned_data):
    data_set = {}
    for suite_name in aligned_data.index.get_level_values(0).unique():
        data_set[suite_name] = {}
        for bm in aligned_data.loc[suite_name].index.get_level_values(0).unique():
            df = aligned_data.loc[suite_name].loc[bm]
            corr_df = df.swaplevel(axis=1)['IPC'].corr()
            if corr_df.min().max() > 0.5 and df.shape[0] > 100:
                data_set[suite_name][bm] = df
        data_set[suite_name] = pd.concat(data_set[suite_name])

    return pd.concat(data_set)
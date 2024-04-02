import os
import pandas as pd

def get_result_tables_from_folder(folder, include_log=True):
    '''
    Parses a folder organized as:
    - folder/
        - <config1>
            - result.txt
            - CPU2017.<id1>.log
            - CPU2017.<id2>.log
            - ...
        - ...
        - <confign>
            - result.txt
            - CPU2017.<id1>.log
            - CPU2017.<id2>.log
            - ...
    Returns three two-level nested dictionaries with first level indexed by configs and
        the second level indexed by benchmarks.
        The dictionaries and values are:
            * scores_table: Each benchmark's score reported in result.txt
            * runtimes_table: Each benchmark's run time reported in result.txt
            * copies_time_table: a list of run times of each copy reported in CPU2017.<id>.log
    '''
    scores_table = {}
    runtimes_table = {}
    copies_time_table = {}
    for setting in sorted(os.listdir(folder)):
        location = os.path.join(folder, setting)
        if os.path.isdir(location):
            results_file = os.path.join(location, 'result.txt')
            scores_table[setting] = {}
            runtimes_table[setting] = {}
            copies_time_table[setting] = {}
            if os.path.exists(results_file):
                with open(results_file) as f:
                    for line in f:
                        if 'CPU2017' not in line:
                            continue
                        log = line.split()
                        bm = log[0].split(':')[-1]
                        rate = log[1]
                        logfile = log[0].split('i')[0].split('/')[-1] + 'log'
                        runtime = log[2]
                        score = log[3]
                        scores_table[setting][bm+rate] = score
                        runtimes_table[setting][bm+rate] = runtime
                        if include_log:
                            with open(os.path.join(location, logfile)) as lf:
                                copies_time_table[setting][bm+rate] = []
                                for logline in lf:
                                    if 'Copy' in logline:
                                        words = logline.split()
                                        assert bm == words[3]
                                        copies_time_table[setting][bm+rate].append(float(words[-1]))
    return scores_table, runtimes_table, copies_time_table


def get_sir17_tables(folder):
    scores_dict, runtimes_dict, copies_time_dict = get_result_tables_from_folder(folder)
    runtimes_table = pd.DataFrame(runtimes_dict).sort_index(axis=1).astype(float)
    runtimes_table = runtimes_table.dropna(axis=1, how='all').groupby(runtimes_table.index.str.split('_',expand=True).get_level_values(1).str[1:].astype(int), group_keys=True).apply(lambda x: x)
    runtimes_table = runtimes_table.rename(index={bm : bm.split('_')[0] for bm in runtimes_table.index.levels[1]})
    scores_table = pd.DataFrame(scores_dict).sort_index(axis=1)#.astype(float)
    scores_table = scores_table.dropna(axis=1, how='all').groupby(scores_table.index.str.split('_',expand=True).get_level_values(1).str[1:].astype(int), group_keys=True).apply(lambda x: x)
    scores_table = scores_table.rename(index={bm : bm.split('_')[0] for bm in scores_table.index.levels[1]})
    execution_times_table = pd.DataFrame(copies_time_dict).unstack().apply(pd.Series, name='runtime')
    execution_times_table = execution_times_table.unstack(0).dropna(how='all', axis=1).swaplevel(axis=1)[scores_table.columns.values]
    execution_times_table = execution_times_table.groupby(execution_times_table.index.str.split('_',expand=True).get_level_values(1).str[1:].astype(int), group_keys=True).apply(lambda x: x)
    execution_times_table = execution_times_table.rename(index={bm : bm.split('_')[0] for bm in execution_times_table.index.levels[1]})
    return runtimes_table, scores_table, execution_times_table

def parse_java_results_log(result_file):
    time_results, utilization_results = [], []
    with open(result_file) as f:
        for line in f:
            if 'elapsed' in line:
                words = line.split()
                bm = words[0][4:]
                if '_' in bm:
                    s = bm.split('_')
                    bm = s[0][:-6] + '_' + s[1][:-5]
                else:
                    bm = bm[:-11]
                runtime = float(words[1])
                assert 'seconds' in line, 'Do not assume this line units is seconds'
                time_results.append((bm, runtime, words[0][:-1]))
            elif 'utilized' in line:
                words = line.split()
                bm = words[0][4:]
                if '_' in bm:
                    s = bm.split('_')
                    bm = s[0][:-6] + '_' + s[1][:-5]
                else:
                    bm = bm[:-11]
                cpus = float(words[5])
                utilization_results.append((bm, cpus))
    runtime_df = pd.DataFrame(time_results, columns=['benchmark', 'runtime', 'logfile'])
    util_df = pd.DataFrame(utilization_results, columns=['benchmark', 'CPU_utilization'])
    return runtime_df, util_df

def parse_gap_results_file(result_file):
    time_results = []
    with open(result_file) as f:
        for line in f:
            if 'Average' in line:
                words = line.split()
                bm = words[0].split('.')[0]
                runtime = float(words[-1])
                time_results.append((bm, runtime, words[0].split(':')[0]))
    runtime_df = pd.DataFrame(time_results, columns=['benchmark', 'runtime', 'logfile'])
    return runtime_df

def parse_ren_new_results_log(result_file):
    time_results, utilization_results = [], []
    with open(result_file) as f:
        for line in f:
            if 'elapsed' in line:
                words = line.split()
                bm = words[0].split('.')[0]
                runtime = float(words[1])
                assert 'seconds' in line, 'Do not assume this line units is seconds'
                time_results.append((bm, runtime, words[0][:-1]))
            elif 'utilized' in line:
                words = line.split()
                bm = words[0].split('.')[0]
                cpus = float(words[5])
                utilization_results.append((bm, cpus))
    runtime_df = pd.DataFrame(time_results, columns=['benchmark', 'runtime', 'logfile'])
    util_df = pd.DataFrame(utilization_results, columns=['benchmark', 'CPU_utilization'])
    return runtime_df, util_df

def get_ren_folder_results(ren_folder):
    iter_results = {}
    for pref in os.listdir(ren_folder):
        iter_results[pref] = {}
        for run_name in os.listdir(os.path.join(ren_folder, pref)):
            run_id = run_name.split('.')[-1]
            run_folder = os.path.join(ren_folder, pref, run_name)
            results_file = os.path.join(run_folder, 'result.txt')
            if os.path.exists(results_file):
                runtime_df, util_df = parse_ren_new_results_log(results_file)
                if runtime_df.empty: continue
                runtime_df, util_df = runtime_df.set_index('benchmark'), util_df.set_index('benchmark')
                iter_results[pref][run_id] = pd.concat([runtime_df, util_df], axis=1)
            else: continue
        
        if len(iter_results[pref]) > 0:
            iter_results[pref] = pd.concat(iter_results[pref])
            newidx = iter_results[pref].index.set_names('runid', level=0)
            iter_results[pref].set_index(newidx, inplace=True)
        else:
            iter_results.pop(pref)
    
    iter_results = pd.concat(iter_results)
    newidx = iter_results.index.set_names('prefetcher', level=0)
    iter_results.set_index(newidx, inplace=True)

    return iter_results
import pandas as pd
import numpy as np
import os
import sys
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from scipy.signal import medfilt

from src.input_parser import get_input_args
from src.create_dataset import get_raw_data
from src.preprocess import generate_run_length

def get_covs(df, phase_colname, target_counter):
    cov = 0
    ccov = 0
    single_phase_cov = df[target_counter].std() / df[target_counter].mean()
    df.loc[:,'virtual'] = (df[phase_colname] != df[phase_colname].shift()) & (df[phase_colname] != df[phase_colname].shift(-1))
    for phase in df[phase_colname].unique():
        phase_df = df[df[phase_colname] == phase]
        cphase_df = phase_df[~phase_df['virtual']]
        if phase_df.shape[0] > 1:
            cov += phase_df.shape[0] * phase_df[target_counter].std() / phase_df[target_counter].mean()
            if cphase_df.shape[0] > 1:
                ccov += single_phase_cov * (phase_df.shape[0] - cphase_df.shape[0]) + (cphase_df.shape[0] * cphase_df[target_counter].std() / cphase_df[target_counter].mean())
            else:
                ccov += single_phase_cov * phase_df.shape[0]
        else:
            cov += phase_df[target_counter].values[0]
            ccov += single_phase_cov
    return single_phase_cov, cov / df.shape[0],  ccov / df.shape[0]

class PhaseTable():
    def __init__(self, features, threshold=1, metric='euclidean', update=None):
        self.table = pd.DataFrame(columns=features)
        self.threshold = threshold
        self.history = {}
        self.update = update
        if metric == 'manhattan':
            self._distance = manhattan_distances
        else:
            self._distance = euclidean_distances
    def is_empty(self):
        return self.table.empty
    def _new_sample(self, sample_row):
        if self.table.empty:
            self.table = self.table.append(sample_row)
            if self.update:
                self.history[0] = self.table.iloc[[0],:]
            return 0
        else:
            distances = self._distance(sample_row, self.table)
            if np.any(distances <= self.threshold):
                label = np.argmin(distances)
                if self.update:
                    self.history[label] = self.history[label].append(sample_row)
                    self.table.iloc[label] = self.history[label].mean()
                return label
            else:
                self.table = self.table.append(sample_row)
                label = len(self.table) - 1
                if self.update:
                    self.history[label] = self.table.iloc[[-1],:]
                return label
    def fit(self, X):
        labels = []
        for idx, row in X.iterrows():
            labels.append(self._new_sample(row.to_frame(name=idx).transpose()))
        return pd.Series(labels, index=X.index, name='Phase')
    def predict(self, X):
        labels = np.argmin(self._distance(X, self.table), axis=1)
        return pd.Series(labels, index=X.index, name='Phase')

class Tier2Phases:
    def __init__(self, N1=10, N2=5, W=20, filter_phases=True):
        self._N1 = N1
        self._N2 = N2
        self._W = W
        # Setting KMeans n_init as instructed in https://github.com/scikit-learn/scikit-learn/discussions/25016
        self._kmeans_l1 = KMeans(N1, max_iter=1000, random_state=42, n_init=10)
        self._kmeans_l2 = KMeans(N2, random_state=42, n_init=10)
        self.background_phase = False
        self._filter_phases = filter_phases
    
    def _get_subphases(self, X):
        return pd.Series(self._kmeans_l1.predict(X), index=X.index, name='Subphases')
    
    def _get_buffers(self, subphases):
        if subphases.index.nlevels == 1:
            buffers = pd.DataFrame(columns=[i for i in range(self._N1)])
            for _, series in list(subphases.groupby(subphases.index // self._W)):
                buffers = pd.concat([buffers, pd.DataFrame.from_records([series.value_counts().to_dict()])], ignore_index=True)
        else:
            per_core_buffers = []
            corenames = subphases.index.get_level_values(0).unique()
            for core in corenames:
                buffers = pd.DataFrame(columns=[i for i in range(self._N1)])
                for _, series in list(subphases.loc[core].groupby(subphases.loc[core].index // self._W)):
                    buffers = pd.concat([buffers, pd.DataFrame.from_records([series.value_counts().to_dict()])], ignore_index=True)
                per_core_buffers.append(buffers)
            buffers = pd.concat(per_core_buffers, keys=corenames)

        return buffers.astype(float).fillna(0)
    
    def _get_phases(self, subphases, buffers):
        if subphases.index.nlevels == 1:
            phase_idx = self._W * (subphases.index // self._W).unique()
            df = pd.DataFrame(subphases)
            df.loc[phase_idx, 'Phases'] = self._kmeans_l2.predict(buffers)
            df = df.ffill().bfill()
        else:
            corenames = subphases.index.get_level_values(0).unique()
            per_core_phases = []
            for core in corenames:
                phase_idx = self._W * (subphases.loc[core].index // self._W).unique()
                df = pd.DataFrame(subphases.loc[core])
                df.loc[phase_idx, 'Phases'] = self._kmeans_l2.predict(buffers.loc[core])
                df = df.ffill().bfill()
                per_core_phases.append(df)
            df = pd.concat(per_core_phases, keys=corenames)
        return df['Phases']
    
    def _filter_phases(self, X, phases):
        unique_phases = phases.unique()
        max_phase = phases.max()
        for phase in unique_phases:
            values = X[phases == phase]
            cmd = values.mean()
            radius = 2 * values.std()
            out_of_radius = euclidean_distances(values, cmd.values.transpose().reshape(1,-1)) > np.linalg.norm(cmd - radius)
            if np.any(out_of_radius):
                phases.loc[values[out_of_radius].index] = max_phase + 1
                self.background_phase = True
        return phases
    
    def fit(self, X):
        self._kmeans_l1.fit(X)
        subphases = self._get_subphases(X)
        buffers = self._get_buffers(subphases)
        self._kmeans_l2.fit(buffers)
        phases = self._get_phases(subphases, buffers).astype(int)
        if self._filter_phases:
            phases = self._filter_phases(X, phases).astype(int)
        return pd.concat([subphases, phases], axis=1, keys=['Subphases', 'Phases'])
    
    def predict(self, X):
        subphases = self._get_subphases(X)
        buffers = self._get_buffers(subphases)
        phases = self._get_phases(subphases, buffers).astype(int)
        if self._filter_phases:
            phases = self._filter_phases(X, phases).astype(int)
        return pd.concat([subphases, phases], axis=1, keys=['Subphases', 'Phases'])

def pre_classification(raw, filter_size, multicore=False):
    if filter_size > 1:
        df = pd.DataFrame({col : medfilt(raw[col], filter_size) for col in raw.columns}, index=raw.index)
    else:
        df = raw
    if multicore:
        # The scaler needs to be the same for each core
        temp_df = df.stack(0, future_stack=True)
        scaler = MinMaxScaler()
        scaler.fit(temp_df)
        corenames = df.columns.get_level_values(0).unique()
        scaled_df = pd.concat(
                [pd.DataFrame(scaler.transform(df[core][temp_df.columns]), df.index, temp_df.columns) for core in corenames],
                axis=1,
                keys=corenames
            )
    else:
        scaled_df = pd.DataFrame(MinMaxScaler().fit_transform(df), df.index, df.columns)
    return scaled_df

def classify(args, dset):
    if args.classifier == 'table':
        cluster = PhaseTable(dset.columns, threshold=args.classifier_threshold, metric=args.distance_metric)
        phases = cluster.fit(dset)
    elif args.classifier == '2kmeans':
        cluster = Tier2Phases(N1=args.N1, N2=args.phase_count, W=args.W, filter_phases=False)
        phases = cluster.fit(dset)['Phases']
    elif args.classifier == 'pcakmeans':
        pca_values = PCA(args.pca, random_state=42).fit_transform(dset)
        cluster = KMeans(args.phase_count, random_state=42, n_init=10).fit(pca_values)
        phases = pd.Series(cluster.predict(pca_values), index=dset.index)
    elif args.classifier == 'gmm':
        cluster = GaussianMixture(args.phase_count, random_state=42).fit(dset)
        phases = pd.Series(cluster.predict(dset), index=dset.index)
    return phases

def main():
    args = get_input_args(sys.argv[0])
    raw = get_raw_data(args)
    if args.multicore_phases:
        assert raw.columns.nlevels == 2, 'multicore_phases option is only valid for multicore data sets'

    scaled_df = pre_classification(raw, args.filter_size, args.multicore_phases)


    data_sets = [scaled_df]
    
    if args.multicore_phases:
        corenames = scaled_df.columns.get_level_values(0).unique()
        if args.multicore_phases == 'local':
            data_sets = []
            for core in corenames:
                data_sets.append(scaled_df[core])
        elif args.multicore_phases == 'local+shared':
            data_sets = [scaled_df.stack(0).swaplevel(0,1).sort_index()]
            corenames = data_sets[0].index.get_level_values(0).unique()

    sets_phases = []

    for dset in data_sets:
        phases = classify(args, dset)
        sets_phases.append(phases)

    print('Metrics:')
    if args.multicore_phases:
        results = {}
        csv = []
        for i, core in enumerate(corenames):
            df = raw[core].copy()
            
            if args.multicore_phases == 'local':
                df.loc[sets_phases[i].index, 'Phase'] = sets_phases[i].values
            elif args.multicore_phases == 'global':
                df['Phase'] = sets_phases[0]
            elif args.multicore_phases == 'local+shared':
                df['Phase'] = sets_phases[0].loc[core]
            
            single_phase_cov, cov,  ccov = get_covs(df, 'Phase', args.input_counters[0])
            rle = generate_run_length(df['Phase'], 0, None, False)
            avg_duration = rle['length'].mean()
            results[core] = {
                'single_phase_cov' : single_phase_cov,
                'cov' : cov,
                'ccov' : ccov,
                'phase_count' : len(phases.unique()),
                'avg_duration' : avg_duration

            }

            csv.append(df[[args.input_counters[0], 'Phase']])
        print(results)
        if args.predictions_csv:
            pd.concat(csv, axis=1, keys=corenames).to_csv(os.path.join(args.results_folder, args.dataset, args.benchmark + '_' + args.classifier + '_' + args.multicore_phases + '.csv'))

    else:
        phases = sets_phases[0]
        df = raw
        df['Phase'] = phases
        single_phase_cov, cov,  ccov = get_covs(df, 'Phase', args.input_counters[0])
        rle = generate_run_length(phases, 0, None, False)
        avg_duration = rle['length'].mean()
        print({
            'single_phase_cov' : single_phase_cov,
            'cov' : cov,
            'ccov' : ccov,
            'phase_count' : len(phases.unique()),
            'avg_duration' : avg_duration
            })
        if args.predictions_csv:
            df[[args.input_counters[0], 'Phase']].to_csv(os.path.join(args.results_folder, args.dataset, args.benchmark + '_' + args.classifier + '.csv'))
    


if __name__ == "__main__":
    main()
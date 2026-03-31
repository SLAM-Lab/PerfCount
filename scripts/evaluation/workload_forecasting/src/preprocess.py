import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, KBinsDiscretizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import itertools
from scipy.signal import medfilt
import os


def read_multicore_golden_phases(bm, data_set, golden_name, phase_type):
	folder = os.path.join('golden_phases', 'set' + str(data_set))
	filename = '_'.join([bm, golden_name, phase_type + '.csv'])
	phases = pd.read_csv(
			os.path.join(folder, filename),
			index_col=0,
			header=[0,1]
		).rename(columns={'L1_cluster' : 'K'}, level=1).swaplevel(0,1,axis=1)['K']
	return phases



def read_golden_phases(bm, data_set, use_multi_level_filtered_clusters, use_manual_label, other_golden_file):
	folder = os.path.join('golden_phases', 'set' + str(data_set))
	if other_golden_file:
		level = '2' if other_golden_file == 'mlc' else '1'
		cluster = 'L'+level+'_cluster'
		phases = pd.read_csv(
				os.path.join(folder, bm + '_'+other_golden_file+'.csv'),
				index_col=0,
				header=0
			).loc[:,[cluster]].rename(columns={cluster : 'K'})
	elif use_multi_level_filtered_clusters:
		level = use_multi_level_filtered_clusters
		cluster = 'L'+level+'_cluster'
		phases = pd.read_csv(
				os.path.join(folder, bm + '_multi_level_filtered_clusters.csv'),
				index_col=0,
				header=0
			).loc[:,[cluster]].rename(columns={cluster : 'K'})
	elif use_manual_label:
		level = use_manual_label
		cluster = 'L'+level+'_cluster'
		phases = pd.read_csv(
				os.path.join(folder, bm + '_manual_label.csv'),
				index_col=0,
				header=0
			).loc[:,[cluster]].rename(columns={cluster : 'K'})
	else:
		phases = pd.read_csv(
				os.path.join(folder, bm + '_phases.csv'),
				index_col=0,
				header=0
			).loc[:,'K']
	class_count = len(phases['K'].unique())
	return phases, class_count


def get_clustered_series_and_clusters(data, k):
	# Scale data
	scaler = MinMaxScaler()
	scaler.fit(data)
	scaled_data = scaler.transform(data)
	kmeans = KMeans(k)
	kmeans.fit(scaled_data)
	series = kmeans.predict(scaled_data)
	clusters = scaler.inverse_transform(kmeans.cluster_centers_)
	return series, clusters

def get_clustered_series_and_transformers(data, k):
	# Scale data
	scaler = MinMaxScaler()
	scaler.fit(data)
	scaled_data = scaler.transform(data)
	kmeans = KMeans(k)
	kmeans.fit(scaled_data)
	series = kmeans.predict(scaled_data)
	clusters = scaler.inverse_transform(kmeans.cluster_centers_)
	return series, clusters, scaler, kmeans

def cluster_scaled_data_get_transformer(data, k):
	kmeans = KMeans(k, random_state=42)
	kmeans.fit(data)
	series = kmeans.predict(data)
	clusters = kmeans.cluster_centers_
	return series, clusters, kmeans

def filter_data(data, filter_name, filter_size):
	if filter_name == "median":
		return medfilt(data, filter_size)
	return data

def discretize_time_series(series, strategy, k):
	d = KBinsDiscretizer(k, "ordinal", strategy)
	d.fit(series)
	return d, d.transform(series)

def get_scaler(data, scaler_name="minmax"):
	if scaler_name == "minmax":
		scaler = MinMaxScaler(feature_range=(-1, 1))
	elif scaler_name == "standard":
		scaler = StandardScaler()
	elif scaler_name == "minmax01":
		scaler = MinMaxScaler(feature_range=(0, 1))
	scaler = scaler.fit(data)
	return scaler

def apply_scaler(scaler, data):
	# transform train
	# data = data.reshape(data.shape[0], data.shape[1])
	data_scaled = scaler.transform(data)
	return data_scaled

def discretize_univariate_series(series, bins, strategy, k):
	def get_labeled_bins_center(bins):
		# {label : center}
		centers = {}
		for i in range(len(bins) - 1):
			centers[i] = (bins[i] + bins[i+1]) / 2
		return centers
	if strategy == "const_bins":
		classes_mean = get_labeled_bins_center(bins)
		discrete_series = pd.cut(series, bins, labels=range(len(bins)-1), include_lowest=True).astype("int")
	else:
		discretizer, discrete_series = discretize_time_series(series.values.reshape(-1,1), strategy, k)
		discrete_series = pd.Series(discrete_series.reshape(-1,))
		classes_mean = get_labeled_bins_center(discretizer.bin_edges_[0])
	df = series
	index = pd.Index(discrete_series)
	for bucket in discrete_series.unique():
		loc = index.get_loc(bucket)
# 		# index.get_loc returns loc : int if unique index, slice if monotonic index, else mask
		if isinstance(loc, int):
			classes_mean[bucket] = df[loc + df.index.values[0]]
		else:
			classes_mean[bucket] = df[loc].mean()
		assert(isinstance(classes_mean[bucket], (int, float)))
	return discrete_series, classes_mean

def get_pca_transform(data, pca_var, fit_index=0):
		pca_obj = PCA(pca_var, random_state=42)
		# First normalize data
		if fit_index != 0:
			# Fit training data only
			scaler = get_scaler(data.iloc[0:fit_index, :], "standard")
			scaled_data = apply_scaler(scaler, data)			
			pca_obj = pca_obj.fit(scaled_data[0:fit_index, :])
		else:
			scaler = get_scaler(data, "standard")
			scaled_data = apply_scaler(scaler, data)
			pca_obj = pca_obj.fit(scaled_data)
		return pca_obj.transform(scaled_data)

def generate_run_length(discrete_series, max_length, classes_mean, shift_length=True):
	# Run Length Encoding
	rle_df = pd.DataFrame(columns=["length", "bin_label"])
	items_count = 0
	groups_list = [(list(group), int(name)) for name, group in itertools.groupby(discrete_series)]
	for group, name in groups_list:
		group_count = len(group)
		if max_length > 0:
			while group_count > max_length:
				index_value = discrete_series.index.values[items_count]
				rle_df.loc[index_value, "bin_label"] = name
				if shift_length:
					rle_df.loc[index_value+max_length, "length"] = max_length - 1
				else:
					rle_df.loc[index_value, "length"] = max_length - 1
				group_count -= max_length
				items_count += max_length
		index_value = discrete_series.index.values[items_count]
		rle_df.loc[index_value, "bin_label"] = name
		if shift_length:
			rle_df.loc[index_value+group_count, "length"] = group_count - 1
		else:
			rle_df.loc[index_value, "length"] = group_count - 1
		items_count += group_count
	
	rle_df = rle_df.dropna(subset=['bin_label']).astype(float).bfill().astype({"bin_label": "int"})
	return rle_df

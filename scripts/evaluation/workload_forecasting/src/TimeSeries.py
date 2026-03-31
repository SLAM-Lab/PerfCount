import os
import pandas as pd
import src.preprocess as pp
import numpy as np
from sklearn.metrics import pairwise_distances_argmin


class TimeSeries:
	def __init__(self, frame):
		self.data=frame
	def get_differenced_series(self, d=1):
		return TimeSeries((self.data.shift(-d) - self.data).dropna())
	def get_supervised_series(self, x, y):
		inputs = self.data.loc[:,x]
		outputs = self.data.loc[:,y]
		return TimeSeries(pd.concat(inputs, output.shif(-1)).dropna())
	def get_discrete_univariate_series(self, bins):
		assert len(self.data.columns) == 1
		return TimeSeries(pd.cut(self.data, bins, labels=range(len(bins)-1), include_lowest=True).astype("int"))

class TimeSeriesTransforms:
	def __init__(self, frame):
		columns = self.__class__._new_transform_columns("original", frame.columns.values)
		self.df = pd.DataFrame(frame.values, index=frame.index, columns=columns)
	@staticmethod
	def _new_transform_columns(transform, column_names):
		return pd.MultiIndex.from_product(
				[[transform], column_names],
				names = ["transform", "feature"]
			)
	def _add_columns_data(self, values, columns_names, index):
		self.df = pd.concat(
			[self.df, pd.DataFrame(values, index=index, columns=columns_names)],
			axis=1
		)

	def apply_filter(self, filter_name, filter_size, transform):
		data = self.df[transform].dropna()
		columns = self.__class__._new_transform_columns("filter", data.columns.values)
		fdata = pd.DataFrame(	{col : pp.filter_data(data[col].values, filter_name, filter_size) for col in data.columns.values},
								index = data.index
							)
		self._add_columns_data(fdata.values, columns, data.index)

	def apply_pca(self, transform, fit_split, pca_var):
		data = self.df[transform].dropna()
		if fit_split > 0 and fit_split < 100:
			# Fit with only training data
			fit_index = int(data.shape[0] * fit_split / 100)
		else: fit_index = 0
		pca_data = pp.get_pca_transform(data, pca_var, fit_index)
		n_components = pca_data.shape[1]
		print('[TimeSeries.py] PCA component count:', n_components)
		names = ["comp_" + str(i) for i in range(n_components)]
		columns = self.__class__._new_transform_columns("pca", names)
		self._add_columns_data(pca_data, columns, data.index)

	def add_timesteps(self, input_transform, timesteps):
		ref_data = self.df[input_transform].dropna()
		data = pd.DataFrame(ref_data.values, columns=["t_" + name for name in ref_data.columns.values], index=ref_data.index)
		for i in range(1, timesteps):
			data = pd.concat( [data,
								pd.DataFrame(ref_data.shift(i).values, columns=['t'+str(i)+'_'+name for name in ref_data.columns.values], index=ref_data.index)
							], axis=1 )
			if len(data.dropna()) < timesteps:
				# Add 0-padding
				data = data.fillna(0)
		columns = self.__class__._new_transform_columns("timesteps", data.columns.values)
		self._add_columns_data(data.values, columns, data.index)
		
	def to_supervised(self, args):
		steps = 1
		df = self.df
		x_cols = df.columns.get_loc(self.it)
		y_cols = df.columns.get_loc( (self.tt, self.tf) )
		self.X = pd.DataFrame(df.iloc[:, x_cols].droplevel(0, 1))
		self.y = pd.DataFrame(df.droplevel(0, 1).iloc[:, y_cols].shift(-steps))
		# Add multiple output steps when forecast horizon > 1
		
		y_out = list(self.y.columns.get_level_values(-1))
		for i in range(1, args.forecast_horizon):
			self.y = pd.concat( [self.y, 
								pd.DataFrame(df.iloc[:, y_cols].shift(-(steps+i)).values,
												index=df.index,
												columns=[col + '+' + str(steps+i) for col in y_out])
								], axis=1
							)
		xcolumns = self.__class__._new_transform_columns("supervised_inputs", self.X.columns.values)
		ycolumns = self.__class__._new_transform_columns("supervised_output", self.y.columns.values)
		self._add_columns_data(self.X.values, xcolumns, self.X.index)
		self._add_columns_data(self.y.values, ycolumns, self.y.index)
		drop = self.X.isna().any(axis=1) | self.y.isna().any(axis=1)
		self.X = self.X[~drop]
		self.y = self.y[~drop]

	def set_transforms(self, args):
		# it: input transform
		# tt: target transform
		# tf: target feature
		self.it = self.tt = "original"
		self.tf = args.input_counters[0]
		# Step 1: Resample
		## Skip
		# Step 2: Filtering
		if args.filter != "none" and args.filter_size > 1:
			self.apply_filter(args.filter, args.filter_size, self.it)
			self.it = self.tt = "filter"
		# Step 3: Differencing
		## Skip
		# Step 4: Discretize
		## Skip
		# Step 5: Run Length Encoding
		## Skip
		# Step 6: PCA
		if args.pca:
			self.apply_pca(self.it, args.first_split, args.pca)
			self.it = "pca"
		# Step 7: Extend timesteps for input
		if args.timesteps > 1 and args.model != "mp":
			self.add_timesteps(self.it, args.timesteps)
			self.it = "timesteps"
		# Step 8: Is it a phase-change binary predictor?
		## Skip
		# Step 9: Time Series to supervised
		self.to_supervised(args)

	def invert_predicted_transforms(self, predictions):
		df = predictions.copy()
		columns = self.__class__._new_transform_columns("predictions", df.columns.values)
		self._add_columns_data(df.values, columns, df.index)
		
		columns = self.__class__._new_transform_columns("inverted_predictions", df.columns.values)
		self._add_columns_data(df.values, columns, df.index)
		return df

	def set_train_test_split(self, args, fit_index=None):
		X = self.X
		y = self.y
		if not fit_index:
			train_split = int(self.df['original'].dropna().shape[0] * args.train_size / 100)
			fit_index = self.df['original'].dropna().index[train_split]
		use_local_split = False
		if fit_index < X.index.values[0]:
			use_local_split = True
			print('[src/TimeSeries.py] WARNING: There is no training data, will use local split')
		elif X.loc[X.index >= fit_index].empty:
			use_local_split = True
			print('[src/TimeSeries.py] WARNING: There is no test data, will use local split')
		if use_local_split:
			local_split = int(args.train_size * len(X) / 100)
			local_split_idx = X.index[local_split]
			self.X_train = X.loc[X.index < local_split_idx]
			self.y_train = y.loc[y.index < local_split_idx]

			self.X_test = X.loc[X.index >= local_split_idx]
			self.y_test = y.loc[y.index >= local_split_idx]
		else:
			self.X_train, self.X_test = X.loc[X.index < fit_index], X.loc[X.index >= fit_index]
			self.y_train, self.y_test = y.loc[self.X_train.index], y.loc[self.X_test.index]

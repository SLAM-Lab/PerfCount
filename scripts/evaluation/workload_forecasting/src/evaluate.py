from sklearn.metrics import (confusion_matrix, accuracy_score, mean_squared_error, explained_variance_score,
		balanced_accuracy_score, roc_curve, r2_score)
from math import sqrt
import pandas as pd
import numpy as np
import os

def get_rmse(validation_set, predictions):
	return sqrt(mean_squared_error(validation_set, predictions))

def get_explained_variance(y_true, y_pred):
	return explained_variance_score(y_true, y_pred)

def get_relative_errors(validation_set, predictions):
	# Relative percentage error
	rpe = 100*(validation_set - predictions)/validation_set
	# Mean Percentage Error
	mpe = np.sum(rpe) / rpe.size
	# Mean Absolute Percentage Error
	mape = np.sum(np.absolute(rpe)) / rpe.size
	return mpe, mape

def get_labeled_changes(predictions, raw_inputs, change_threshold):
	# A POSITIVE is a change of 1 CPI unit or 10% if previous value vs NEGATIVE (no change)
	labels = np.zeros(len(predictions))
	up_prediction = np.logical_or( (predictions - raw_inputs) >= 1, ((predictions - raw_inputs) / raw_inputs) > change_threshold)
	down_prediction = np.logical_or( (predictions - raw_inputs) <= -1, ((predictions - raw_inputs) / raw_inputs) < -change_threshold)
	labels[up_prediction] = 1
	labels[down_prediction] = 2
	return labels

def get_accuracies(y_true, y_pred):
	acc, bal_acc = [], []
	for i in range(y_pred.shape[1]):
		acc.append(accuracy_score(y_true[:,i], y_pred[:,i]))
		bal_acc.append(balanced_accuracy_score(y_true[:,i], y_pred[:,i]))
	return acc, bal_acc


def get_regression_metrics(y_true, y_hat, column_count):
	metrics = {}
	# RMSE
	rmse = get_rmse(y_true, y_hat)
	# Relative errors
	mpe, mape = get_relative_errors(y_true, y_hat)
	max_error = (y_true - y_hat).max()
	r2 = r2_score(y_true, y_hat)
	# Variance
	var = get_explained_variance(y_true, y_hat)

	metrics['rmse'] = rmse
	metrics['max'] = max_error
	metrics['mpe'] = mpe
	metrics['exp_var'] = var
	metrics['mape'] = mape
	metrics['r2_score'] = r2

	if column_count > 1:
		rmse_list = list()
		mape_list = list()
		r2_list = list()
		for i in range(column_count):
			rmse_list.append(get_rmse(y_true[:,i], y_hat[:,i]))
			_, n_mape = get_relative_errors(y_true[:,i], y_hat[:,i])
			mape_list.append(n_mape)
			r2_list.append(r2_score(y_true[:,i], y_hat[:,i]))
		metrics['rmse_per_step'] = ':'.join(str(x) for x in rmse_list)
		metrics['mape_per_step'] = ':'.join(str(x) for x in mape_list)
		metrics['r2_per_step'] = ':'.join(str(x) for x in r2_list)

	return metrics


def get_validation_set(transform, timeseries, target, lag, forecast_horizon, phases=pd.DataFrame()):
	if phases.empty:
		validation_set_df = pd.DataFrame(timeseries.df[transform][target].shift(-lag))
		for i in range(lag, forecast_horizon + lag - 1):
			validation_set_df.loc[:,target+'+'+str(i+1)] = timeseries.df[transform][target].shift(-(i+1))
	else:
		df = pd.DataFrame(timeseries.df[transform][target])
		separate_series = { p : cluster_df for p, cluster_df in df.groupby(phases)}
		for phase in separate_series.keys():
			df = separate_series[phase]
			validation_df = separate_series[phase].shift(-lag)
			for i in range(lag, forecast_horizon + lag - 1):
				validation_df.loc[:,target+'+'+str(i+1)] = df.shift(-(i+1))
			separate_series[phase] = validation_df
		validation_set_df = pd.concat([df for df in list(separate_series.values())]).sort_index()
	return validation_set_df


def print_all_results(args, timeseries, predictions_df, y_hat, y_test, target, print_metrics=True,
						phases=pd.DataFrame()):
	RESULTS_FOLDER 			= args.results_folder
	create_predictions_csv	= args.predictions_csv
	
	validation_set_df = get_validation_set("original", timeseries, target, 1,
											args.forecast_horizon, phases).loc[predictions_df.index,:]

	save_df_keys = ["True", "Prediction"]
	val_df = pd.concat( [validation_set_df, predictions_df], axis=1, keys=save_df_keys).dropna()
	save_df_list = [validation_set_df.shift(1), predictions_df.shift(1)]
	
	predictions = val_df["Prediction"].values
	validation_set = val_df["True"].values

	results = {}
	true_data_sets = [validation_set]
	true_data_names = ['']

	if args.filter != "none" and args.filter_size > 1:
		filter_val_set_df = get_validation_set("filter", timeseries, target, 1, args.forecast_horizon).loc[val_df.index, :]
		filter_val_set = filter_val_set_df.values
		save_df_list.append(filter_val_set_df.shift(1))
		save_df_keys.append("Filter")
		true_data_sets.append(filter_val_set)
		true_data_names.append('filter_')

	for true_values, true_name in zip(true_data_sets, true_data_names):
		metrics = get_regression_metrics(true_values, predictions, args.forecast_horizon)
		for key in metrics.keys():
			results[true_name + key] = metrics[key]

	raw_inputs = timeseries.df["original"][target].loc[val_df.index].values

	file_sfx = args.benchmark+"_"+args.name

	# Print results
	if create_predictions_csv:
		predictions_csv = pd.concat(save_df_list, axis=1, keys=save_df_keys)
		predictions_csv.to_csv(os.path.join(RESULTS_FOLDER, file_sfx+".csv"))
	# if args.transforms_csv:
	# 	timeseries.df.loc[val_df.index].to_csv(os.path.join(RESULTS_FOLDER, file_sfx+"_transforms.csv"))
	if print_metrics:
		print("Metrics:")
		print(results)
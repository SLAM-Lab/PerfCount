import argparse

def set_experiment_args(parser):
	parser.add_argument('--benchmark',			required=True)
	parser.add_argument('--dataset',			required=True,	help='Name of the dataset in the Data folder')
	parser.add_argument('--predictions_csv',	action='store_true',	help='Save predictions in csv format')
	parser.add_argument('--train_size',			type=int,	default=70)
	parser.add_argument('--results_folder',		default='results',	help='Folder to save results')
	parser.add_argument('--name',				default='', help='Experiment name used for results filenames')
	return parser

def set_preprocess_args(parser):
	parser.add_argument('--input_counters',		default=['CPI'],	nargs='+',	help='List of model inputs. '\
												'The first counter in the list will be used as the ouput of '\
												'forecasting models')
	parser.add_argument('--pca',				help='Use PCA to reduce the dimensionality of the data. Use ' \
												'integer values for the number of components or a fraction for' \
												' the percentage of variance')
	parser.add_argument('--start_drop_count',	type=int,	default=0,	help='Number of samples to drop at the'\
												' beginning of the training set')
	parser.add_argument('--end_drop_count',		type=int, default=0, 	help='Number of samples to drop at the'\
												' end of the training set')
	parser.add_argument('--scaler',				default='minmax', choices=['minmax', 'standard', 'none', 'minmax01'])
	parser.add_argument('--filter',				default='none', choices=['median', 'none'])
	parser.add_argument('--filter_size',		type=int, default=3)
	return parser

def supervised_model_args(parser):
	parser.add_argument("--batch_size",			type = int, default = 16)
	parser.add_argument("--epochs",				type = int, default = 100)
	parser.add_argument("--neurons",			type = int, default = 16)
	parser.add_argument("--stateless",			action = "store_true")
	parser.add_argument("--model",				default = "lstm", choices = ["lstm", "stacked_lstm", "transformer", "mlp", "svm", "mp", 'dt', 'lr'])
	parser.add_argument("--dense_hidden_layers", type = int, default = [50, 50], nargs = '+')
	parser.add_argument("--early_stopping",		action = "store_true")
	parser.add_argument("--patience",			type = int, default = 5)
	parser.add_argument("--loss_function",		default = "mse", choices = ["mse", "mean_squared_error", "mae", "mean_absolute_error", "mean_absolute_percentage_error", "mean_squared_logarithmic_error"])
	parser.add_argument("--regression_activation",	default='linear', choices=['relu', 'linear', 'tanh', 'sigmoid'])
	parser.add_argument("--optimizer",			default = "adam", choices = ["sgd", "RMSprop", "adagrad", "adadelta", "adam", "adamax", "nadam"])
	parser.add_argument("--svm_kernel",			default = "rbf", choices = ["linear", "poly", "rbf", "sigmoid"])
	parser.add_argument("--svm_regularization",	type = float, default = 1.0)
	parser.add_argument("--svm_epsilon",		type = float, default = 0.1)
	parser.add_argument("--max_iter",			type = int, default = -1)
	parser.add_argument("--stacked_layers",		type = int, default = 2)
	parser.add_argument("--tree_max_depth",		type = int, default = 3)
	# Transformer-specific parameters (optimized via hyperparameter sweep)
	parser.add_argument("--num_heads",			type = int, default = 2, help="Number of attention heads for transformer (optimal: 2)")
	parser.add_argument("--dropout_rate",		type = float, default = 0.2, help="Dropout rate for transformer (optimal: 0.2)")
	return parser
	

def get_input_args(caller_file, input_args=None):
	parser = argparse.ArgumentParser()
	parser = set_experiment_args(parser)
	parser = set_preprocess_args(parser)
	
	if 'classify' in caller_file:
		parser.add_argument('--phase_count',			type=int, required=True)
		parser.add_argument('--classifier',				choices=['table', '2kmeans', 'pcakmeans', 'gmm'])
		parser.add_argument('--classifier_threshold',	type=float, default=1)
		parser.add_argument('--distance_metric',		choices=['euclidean', 'manhattan'])
		parser.add_argument('--W',						type=int,	default=100)
		parser.add_argument('--N1',						type=int,	default=10)
		parser.add_argument('--multicore_phases', 		choices=['local', 'global', 'local+shared'])
	
	if 'forecasting' in caller_file:
		parser.add_argument("--timesteps",			type = int, default=1, help='Number of input timesteps')
		parser.add_argument("--forecast_horizon",	type = int, default=1, help="Number of output timesteps")
		# parser.add_argument("--no_deltas",			action = "store_true", help= "Do not use relative counter changes")
		parser = supervised_model_args(parser)

	if input_args:
		args = parser.parse_args(args=input_args)
	else:
		args = parser.parse_args()

	# Special argument types:
	# PCA
	if args.pca:
		if args.pca.isdigit():
			args.pca = int(args.pca)
		else:
			args.pca = float(args.pca)
	return args



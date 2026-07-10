import pandas as pd
import numpy as np
import os
import sys
import copy
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from scipy.signal import medfilt
import argparse

from src.input_parser import get_input_args
from src.create_dataset import (
    get_raw_data,
    reshape_with_batch_size,
    get_separate_time_series_splits,
)
from src.preprocess import generate_run_length
from src.evaluate import print_all_results
import src.Predictor as pred

def main():
    # 1. Intercept custom autoregressive arguments before the main parser sees them
    custom_parser = argparse.ArgumentParser(add_help=False)
    custom_parser.add_argument('--target_counters', nargs='+', default=None)
    custom_parser.add_argument('--autoregressive_steps', type=int, default=0)
    custom_args, remaining_argv = custom_parser.parse_known_args()

    # Pass the clean remaining arguments to your original parser
    sys.argv = [sys.argv[0]] + remaining_argv
    args = get_input_args(sys.argv[0])
    
    data = get_raw_data(args)
    if data is None or len(data) == 0:
        raise ValueError(f"No data loaded for benchmark {args.benchmark}")

    # =========================================================
    # ORIGINAL WORKFLOW (If not running Autoregressive)
    # =========================================================
    if custom_args.autoregressive_steps <= 1:
        X_train, y_train, X_test, y_test, train_timeseries, test_timeseries = get_separate_time_series_splits(args, data)
        target = args.input_counters[0]
        
        min_observation = train_timeseries.df["original"][target].min()
        max_observation = train_timeseries.df["original"][target].max()
         
        X_train, y_train, X_test, y_test = reshape_with_batch_size(X_train, y_train, X_test, y_test, args.batch_size)
        train_data = pred.PredictorInputs(args, X_train, y_train)
        test_data = pred.PredictorInputs(args, X_test, y_test, train_data.in_scaler, train_data.out_scaler, batch_padding=True)

        predictor = pred.SerialPredictor(args, train_data.X)
        predictions = predictor.train_predict(args, train_data, test_data)
        predictor.save_model(args)

        test_data.add_predictions(args, predictions)
        predictions_df = test_timeseries.invert_predicted_transforms(test_data.y_hat)
        
        print_all_results(args, test_timeseries, predictions_df, test_data.y_hat, y_test, target)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)

    # =========================================================
    # AUTOREGRESSIVE WORKFLOW (Chaining Models)
    # =========================================================
    targets = custom_args.target_counters if custom_args.target_counters else args.input_counters
    steps = custom_args.autoregressive_steps
    original_input_counters = list(args.input_counters)

    trained_models = {}
    test_scalers = {}
    model_input_cols = {}
    
    # 1. Train a separate base model for each of the 4 targets
    for target in targets:
        # Trick create_dataset by rotating the target to index 0
        current_inputs = [target] + [c for c in original_input_counters if c != target]
        args.input_counters = current_inputs
        model_input_cols[target] = list(current_inputs)
        
        X_train, y_train, X_test, y_test, train_ts, test_ts = get_separate_time_series_splits(args, data)
        X_train, y_train, X_test, y_test = reshape_with_batch_size(X_train, y_train, X_test, y_test, args.batch_size)
        
        train_data = pred.PredictorInputs(args, X_train, y_train)
        test_data = pred.PredictorInputs(args, X_test, y_test, train_data.in_scaler, train_data.out_scaler, batch_padding=True)
        
        predictor = pred.SerialPredictor(args, train_data.X)
        _ = predictor.train_predict(args, train_data, test_data) 
        
        trained_models[target] = predictor
        test_scalers[target] = (test_data, test_ts, y_test)

    # Reset args back to the main CPU cycle tracking
    args.input_counters = original_input_counters
    main_target = original_input_counters[0] 
    
    main_test_data, main_test_ts, main_y_test = test_scalers[main_target]
    current_X_test = np.copy(main_test_data.X)
    
    # 2. Autoregressive Chaining Loop
    for step in range(steps):
        step_predictions = {}
        
        for target in targets:
            predictor = trained_models[target]
            expected_cols = model_input_cols[target]
            
            # Reorder columns dynamically to match what this specific model trained on
            col_indices = [original_input_counters.index(c) for c in expected_cols]
            if len(current_X_test.shape) == 2:
                reordered_X = current_X_test[:, col_indices]
            else:
                reordered_X = current_X_test[:, :, col_indices]
            
# Execute Prediction (in scaled space)
            # Find the actual model attribute dynamically (Keras vs Scikit-Learn naming)
            underlying_model = None
            for attr in ['model', 'predictor', 'regressor', 'estimator', 'clf']:
                if hasattr(predictor, attr):
                    underlying_model = getattr(predictor, attr)
                    break
            if underlying_model is None:
                underlying_model = predictor # Fallback to the wrapper itself
                
            # Execute prediction (Keras accepts batch_size, sklearn does not)
            try:
                preds = underlying_model.predict(reordered_X, batch_size=args.batch_size)
            except TypeError:
                preds = underlying_model.predict(reordered_X)
                
            step_predictions[target] = preds.flatten() if isinstance(preds, np.ndarray) else preds
            
        if step == steps - 1:
            final_predictions = step_predictions[main_target]
            break
            
        # 3. Inject predictions back into current_X_test for the next loop
        for feature_idx, feature_name in enumerate(original_input_counters):
            if feature_name in step_predictions:
                if len(current_X_test.shape) == 2:
                    current_X_test[:, feature_idx] = step_predictions[feature_name]
                elif len(current_X_test.shape) == 3:
                    if current_X_test.shape[1] == 1:
                        current_X_test[:, 0, feature_idx] = step_predictions[feature_name]
                    else:
                        current_X_test[:, :-1, feature_idx] = current_X_test[:, 1:, feature_idx]
                        current_X_test[:, -1, feature_idx] = step_predictions[feature_name]
                    
    # 4. Final Evaluation using standard pipeline
    main_test_data.add_predictions(args, final_predictions)
    predictions_df = main_test_ts.invert_predicted_transforms(main_test_data.y_hat)
    
    print_all_results(args, main_test_ts, predictions_df, main_test_data.y_hat, main_y_test, main_target)

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)

if __name__ == "__main__":
    main()
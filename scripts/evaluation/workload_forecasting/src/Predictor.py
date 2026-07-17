import pandas as pd
import numpy as np
import os
from keras.callbacks import EarlyStopping, ModelCheckpoint
import src.preprocess as pp
from src.my_keras_models import get_model, reset_model_states
from sklearn.svm import SVR, SVC, LinearSVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.tree import DecisionTreeRegressor

# Set random seeds for reproducibility. WF_SEED overrides the default (still 42)
# so a single fit can be re-run under a different initialization.
RANDOM_SEED = int(os.environ.get("WF_SEED", "42"))
np.random.seed(RANDOM_SEED)

# Set TensorFlow/Keras random seeds for reproducibility
try:
    import tensorflow as tf
    tf.random.set_seed(RANDOM_SEED)
    # Set operations to be deterministic (may impact performance)
    # This may not be available in older TensorFlow versions
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        # Deterministic ops not available or already enabled
        pass
except ImportError:
    # TensorFlow not available
    pass

# Set Python random seed
import random
random.seed(RANDOM_SEED)

class PredictorInputs:
    def __init__(self, args, X, y, in_scaler=None, out_scaler=None, batch_padding=False):
        self.in_scaler = in_scaler
        self.out_scaler = out_scaler
        self.X_df, self.y_df = X.copy(), y.copy()
        self.pad_rows = 0

        if batch_padding and ('lstm' in args.model or 'transformer' in args.model or 'mlp' in args.model):
            self.pad_rows = args.batch_size - (X.shape[0] % args.batch_size) if args.batch_size > 1 else 0
            if self.pad_rows > 0:
                xpad = pd.DataFrame(np.zeros((self.pad_rows, X.shape[1])), columns=X.columns)
                X = pd.concat([X, xpad], ignore_index=True)
                ypad = pd.DataFrame(np.zeros((self.pad_rows, y.shape[1])), columns=y.columns)
                y = pd.concat([y, ypad], ignore_index=True)

        # # --------------------------------------------------- SHAPE INPUT DATA --------------------------------------------------- # #
        # Scalar inputs only
        self.X, self.in_scaler = self.__class__._scale_data(X.values, args.scaler, self.in_scaler)
        self.X = self.__class__._shape_continuous_inputs(args, self.X)

        # # --------------------------------------- SHAPE OUTPUT DATA --------------------------------------- # #
        # For DT/SVM with horizon=1, select only first column before scaling
        if (args.model == "svm" or args.model == 'dt') and args.forecast_horizon == 1:
            # Take only first column if multiple columns exist
            if y.values.shape[1] > 1:
                y_for_scaling = y.iloc[:, 0:1]  # Keep as 2D with single column
            else:
                y_for_scaling = y
            self.y, self.out_scaler = self.__class__._scale_data(y_for_scaling.values, args.scaler, self.out_scaler)
            # Now reshape to 1D for sklearn
            self.y = self.y.reshape(self.y.shape[0],)
        else:
            # Normal case: scale all columns
            self.y, self.out_scaler = self.__class__._scale_data(y.values, args.scaler, self.out_scaler)

    def _scale_input_output(self, args, X, y):
        # Scale both X and Y
        scaler_name = args.scaler
        scaled_X, self.in_scaler = self.__class__._scale_data(X.values, scaler_name, self.in_scaler)
        scaled_y, self.out_scaler = self.__class__._scale_data(y.values, scaler_name, self.out_scaler)
        return scaled_X, scaled_y
    
    @staticmethod
    def _scale_data(data, scaler_name, scaler=None):
        if scaler_name != 'none':
            if not scaler:
                scaler = pp.get_scaler(data, scaler_name)
            scaled_data = pp.apply_scaler(scaler, data)
            return scaled_data, scaler
        return data, None

    @staticmethod
    def _shape_continuous_inputs(args, X):
        if "lstm" in args.model or "transformer" in args.model:
            X = X.reshape(X.shape[0], args.timesteps, int(X.shape[1] / args.timesteps))
        return X
    
    def add_predictions(self, args, y_hat):
        if self.pad_rows > 0:
            self.y_hat = y_hat[:-self.pad_rows]
        else:
            self.y_hat = y_hat
        scaler_name = args.scaler
        if scaler_name != "none":
            # Ensure predictions are 2D for inverse_transform
            if len(self.y_hat.shape) == 1:
                self.y_hat = self.y_hat.reshape(-1, 1)
            self.y_hat = self.out_scaler.inverse_transform(self.y_hat)
        # For DT/SVM with single output, we may need to match original y_df shape
        if self.y_hat.shape[1] == 1 and len(self.y_df.columns) > 1:
            # Only use the first column of y_df
            self.y_hat = pd.DataFrame(self.y_hat, index=self.y_df.index, columns=[self.y_df.columns[0]])
        else:
            self.y_hat = pd.DataFrame(self.y_hat, index=self.y_df.index, columns=self.y_df.columns)

class Predictor:
    def __init__(self, model):
        self.model = model
    @staticmethod
    def _get_continuous_inputs_count(args, x_shape):
        if len(x_shape) > 1:
            inputs_count = x_shape[1]
        else:
            inputs_count = 1
        if (args.perfect_rle and args.max_run_length > 0) or args.rle:
            inputs_count = 2 # TODO: this assumes RLE always use one counter
        return inputs_count
    @staticmethod
    def _get_model_name_prefix(args):
        model_name = ""
        model_name += args.model
        return model_name

    def train_predict(self, args, train_data, test_data, monitor):
        X, y = train_data.X, train_data.y
        vX, vy = test_data.X, test_data.y
        # Create unique filename with PID and timestamp to avoid collisions in parallel execution
        import time
        unique_id = f"{os.getpid()}_{int(time.time()*1000000)}"
        saved_model_name = os.path.join(args.results_folder,
                                        args.benchmark+'_'+unique_id)
        if args.early_stopping:
            early_stopping_monitor = EarlyStopping(monitor=monitor, patience=args.patience)
            callbacks = [early_stopping_monitor, ModelCheckpoint(filepath=saved_model_name+'.keras', monitor=monitor, save_best_only=True)]
            history = self.model.fit(X, y, validation_data=(vX, vy),
                                    batch_size=args.batch_size,
                                    epochs=args.epochs, shuffle=True,
                                    verbose=0, callbacks=callbacks)
            reset_model_states(self.model)
            self.model.load_weights(saved_model_name+'.keras')
            if early_stopping_monitor.stopped_epoch != 0:
                print("[Predictor.py] Stopped epoch: ", early_stopping_monitor.stopped_epoch)
        else:
            history = self.model.fit(X, y, validation_data=(vX, vy),
                                    batch_size=args.batch_size,
                                    epochs=args.epochs, shuffle=True, verbose=0)
        reset_model_states(self.model)
        output = self.model.predict(vX, batch_size=args.batch_size)
        return output

    def train_predict_svm(self, train_data, test_data):
        self.model.fit(train_data.X, train_data.y.astype(int))
        output = self.model.predict(test_data.X)
        return output

class Regressor(Predictor):
    def __init__(self, args, X):
        # inputs_count = super()._get_continuous_inputs_count(args, X.shape)
        loss = args.loss_function
        dual_activation = None
        class_count = out_steps = args.forecast_horizon
        if "lstm" in args.model or "transformer" in args.model:
            self.input_shape = (args.batch_size, X.shape[1], X.shape[2])
        else:
            self.input_shape = (X.shape[1],)
        self.model_name = super()._get_model_name_prefix(args)
        if args.model == "svm":
            if args.svm_kernel == 'linear':
                self.model = LinearSVR(
                    C=args.svm_regularization,
                    epsilon=args.svm_epsilon,
                    max_iter=args.max_iter,
                )
            else:
                self.model = SVR(
                    kernel=args.svm_kernel,
                    C=args.svm_regularization,
                    epsilon=args.svm_epsilon,
                    max_iter=args.max_iter,
                )
            
            if args.forecast_horizon > 1:
                self.model = MultiOutputRegressor(self.model, n_jobs=1)
        elif args.model == 'dt':
            self.model = DecisionTreeRegressor(max_depth=args.tree_max_depth, random_state=RANDOM_SEED)
            if args.forecast_horizon > 1:
                self.model = MultiOutputRegressor(self.model, n_jobs=1)
        else:
            self.model = get_model(self.model_name, self.input_shape, args.neurons, not args.stateless,
                                            class_count = class_count,
                                            dense_layers = args.dense_hidden_layers,
                                            loss=loss, dual_activation=dual_activation,
                                            optimizer=args.optimizer, stacked_layers=args.stacked_layers,
                                            regression_activation=args.regression_activation,
                                            num_heads=args.num_heads,
                                            dropout_rate=args.dropout_rate)
        super().__init__(self.model)

    def train_predict_svm(self, train_data, test_data):
        X = train_data.X
        y = train_data.y
        self.model.fit(X, y)
        output = self.model.predict(test_data.X)
        return output
    
    def train_predict(self, args, train_data, test_data):
        if args.model == "svm" or args.model == 'dt':
            predictions = self.train_predict_svm(train_data, test_data)
            if args.forecast_horizon == 1:
                predictions = predictions.reshape(-1,1)
        else:
            predictions = super().train_predict(args, train_data, test_data, "val_loss")
        return predictions


class SerialPredictor():
    def __init__(self, args, X):
        self.predictor = Regressor(args,X)
    def train_predict(self, args, train_data, test_data):
        return self.predictor.train_predict(args, train_data, test_data)

    def save_model(self, args):
        """Persist the trained model to args.save_model_path (no extension).

        Keras models (mlp/lstm/transformer/...) are written as ``<path>.keras``;
        scikit-learn models (dt/svm, possibly MultiOutputRegressor-wrapped) as
        ``<path>.joblib``. No-op when --save_model_path is unset.
        """
        path = getattr(args, 'save_model_path', None)
        if not path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        model = self.predictor.model
        if args.model in ('dt', 'svm'):
            import joblib
            joblib.dump(model, path + '.joblib')
        else:
            model.save(path + '.keras')

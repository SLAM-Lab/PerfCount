"""
Model configuration module for cross-platform prediction evaluation.

This module defines all scikit-learn regression models with their
hyperparameter grids.
"""

from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor,
    ExtraTreesRegressor, BaggingRegressor, VotingRegressor, StackingRegressor,
    HistGradientBoostingRegressor
)
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge,
    HuberRegressor, SGDRegressor, PassiveAggressiveRegressor,
    QuantileRegressor, TheilSenRegressor, RANSACRegressor
)
from sklearn.svm import SVR, LinearSVR
from sklearn.neighbors import (
    KNeighborsRegressor, RadiusNeighborsRegressor
)
from sklearn.tree import (
    DecisionTreeRegressor, ExtraTreeRegressor
)
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, ConstantKernel as C
)
from sklearn.kernel_ridge import KernelRidge
from sklearn.isotonic import IsotonicRegression

# Try to import additional boosting libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. "
          "Install with: pip install xgboost")

# Check for Apple ML accelerators (MPS)
try:
    import torch
    MPS_AVAILABLE = torch.backends.mps.is_available()
    if MPS_AVAILABLE:
        print("Apple ML accelerator (MPS) detected and available")
    else:
        MPS_AVAILABLE = False
        print("Apple ML accelerator (MPS) not available")
except ImportError:
    MPS_AVAILABLE = False
    print("PyTorch not available - "
          "Apple ML accelerator support disabled")

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("Warning: CatBoost not available. "
          "Install with: pip install catboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM not available. "
          "Install with: pip install lightgbm")


def get_model_configurations():
    """
    Get all regression models with their hyperparameter grids
    
    Returns:
        dict: Dictionary mapping model names to 
              (model_class, param_grid) tuples
    """
    models = {}
    
    # Linear Models
    models['LinearRegression'] = (LinearRegression(), {})
    
    models['Ridge'] = (Ridge(random_state=42), {
        'alpha': [0.1, 1.0, 10.0, 100.0, 1000.0]
    })
    
    models['Lasso'] = (Lasso(random_state=42), {
        'alpha': [0.01, 0.1, 1.0, 10.0]
    })
    
    models['ElasticNet'] = (ElasticNet(random_state=42), {
        'alpha': [0.01, 0.1, 1.0, 10.0],
        'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
    })
    
    models['BayesianRidge'] = (BayesianRidge(), {
        'alpha_1': [1e-6, 1e-5, 1e-4],
        'alpha_2': [1e-6, 1e-5, 1e-4],
        'lambda_1': [1e-6, 1e-5, 1e-4],
        'lambda_2': [1e-6, 1e-5, 1e-4]
    })
    
    models['HuberRegressor'] = (HuberRegressor(), {
        'epsilon': [1.1, 1.35, 1.5, 2.0],
        'alpha': [0.0001, 0.001, 0.01, 0.1]
    })
    
    models['SGDRegressor'] = (SGDRegressor(random_state=42), {
        'loss': ['squared_error', 'huber', 'epsilon_insensitive'],
        'penalty': ['l2', 'l1', 'elasticnet'],
        'alpha': [0.0001, 0.001, 0.01, 0.1],
        'learning_rate': ['constant', 'optimal', 'invscaling']
    })
    
    models['PassiveAggressiveRegressor'] = (
        PassiveAggressiveRegressor(random_state=42), {
        'C': [0.01, 0.1, 1.0, 10.0],
        'epsilon': [0.01, 0.1, 0.5, 1.0]
    })
    
    models['QuantileRegressor'] = (
        QuantileRegressor(quantile=0.5), {
        'alpha': [0.01, 0.1, 1.0, 10.0],
        'solver': ['highs', 'highs-ds', 'highs-ipm']
    })
    
    models['TheilSenRegressor'] = (
        TheilSenRegressor(random_state=42), {
        'max_subpopulation': [1000, 5000, 10000],
        'n_subsamples': [None, 100, 500]
    })
    
    models['RANSACRegressor'] = (
        RANSACRegressor(random_state=42), {
        'min_samples': [0.5, 0.7, 0.9],
        'residual_threshold': [0.1, 0.5, 1.0],
        'max_trials': [50, 100, 200]
    })
    
    # Support Vector Machines
    models['SVR'] = (SVR(), {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
        'kernel': ['rbf', 'linear', 'poly']
    })
    
    models['LinearSVR'] = (LinearSVR(random_state=42), {
        'C': [0.1, 1, 10, 100],
        'epsilon': [0.01, 0.1, 0.5, 1.0],
        'loss': ['epsilon_insensitive', 'squared_epsilon_insensitive']
    })
    
    # Kernel Methods
    models['KernelRidge'] = (KernelRidge(), {
        'alpha': [0.1, 1.0, 10.0],
        'kernel': ['rbf', 'linear', 'poly'],
        'gamma': [0.001, 0.01, 0.1, 1],
        'degree': [2, 3, 4]
    })
    
    # Neighbors
    models['KNeighborsRegressor'] = (KNeighborsRegressor(), {
        'n_neighbors': [3, 5, 7, 9, 11, 15],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree']
    })
    
    models['RadiusNeighborsRegressor'] = (RadiusNeighborsRegressor(), {
        'radius': [1.0, 2.0, 5.0, 10.0],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree']
    })
    
    # Trees
    models['DecisionTreeRegressor'] = (DecisionTreeRegressor(random_state=42), {
        'max_depth': [None, 5, 10, 15, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['squared_error', 'friedman_mse', 'absolute_error']
    })
    
    models['ExtraTreeRegressor'] = (ExtraTreeRegressor(random_state=42), {
        'max_depth': [None, 5, 10, 15, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    })
    
    # Ensemble Methods
    models['RandomForestRegressor'] = (RandomForestRegressor(random_state=42), {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['squared_error', 'absolute_error']
    })
    
    models['ExtraTreesRegressor'] = (ExtraTreesRegressor(random_state=42), {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    })
    
    models['GradientBoostingRegressor'] = (
        GradientBoostingRegressor(random_state=42), {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7, 10],
        'subsample': [0.8, 0.9, 1.0],
        'criterion': ['squared_error', 'friedman_mse']
    })
    
    models['AdaBoostRegressor'] = (AdaBoostRegressor(random_state=42), {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.5, 1.0],
        'loss': ['linear', 'square', 'exponential']
    })
    
    models['BaggingRegressor'] = (BaggingRegressor(random_state=42), {
        'n_estimators': [10, 50, 100],
        'max_samples': [0.5, 0.7, 0.9, 1.0],
        'max_features': [0.5, 0.7, 0.9, 1.0]
    })
    
    models['HistGradientBoostingRegressor'] = (
        HistGradientBoostingRegressor(random_state=42), {
        'learning_rate': [0.01, 0.1, 0.2],
        'max_iter': [50, 100, 200],
        'max_depth': [3, 5, 7, 10],
        'l2_regularization': [0, 0.1, 1.0]
    })
    
    # Advanced Ensemble Methods (will be added after base models are defined)
    # Note: These require base models to be available
    
    # Neural Networks
    models['MLPRegressor'] = (MLPRegressor(random_state=42, max_iter=1000), {
        'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (100, 100)],
        'activation': ['relu', 'tanh', 'logistic'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'adaptive'],
        'solver': ['adam', 'lbfgs']
    })
    
    # Gaussian Process
    models['GaussianProcessRegressor'] = (
        GaussianProcessRegressor(random_state=42), {
        'kernel': [RBF(), C(1.0) * RBF(), C(1.0) * RBF() + C(1.0)],
        'alpha': [1e-10, 1e-8, 1e-6, 1e-4]
    })
    
    # Specialized Models
    models['IsotonicRegression'] = (IsotonicRegression(), {})
    
    # Add XGBoost if available
    if XGBOOST_AVAILABLE:
        models['XGBRegressor'] = (
            xgb.XGBRegressor(random_state=42, verbosity=0), {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7, 10],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 1],
            'reg_lambda': [1, 1.5, 2]
        })
    
    # Add CatBoost if available
    if CATBOOST_AVAILABLE:
        models['CatBoostRegressor'] = (
            cb.CatBoostRegressor(random_seed=42, verbose=False), {
            'iterations': [50, 100, 200],
            'depth': [3, 5, 7, 10],
            'learning_rate': [0.01, 0.1, 0.2],
            'l2_leaf_reg': [1, 3, 5, 7, 9]
        })
    
    # Add LightGBM if available
    if LIGHTGBM_AVAILABLE:
        models['LGBMRegressor'] = (
            lgb.LGBMRegressor(random_state=42, verbosity=-1), {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7, 10],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 1],
            'reg_lambda': [0, 0.1, 1]
        })
    
    # Add GPU-accelerated models if MPS is available
    if MPS_AVAILABLE:
        try:
            # Add PyTorch-based neural network with MPS support
            from torch import nn
            import torch.nn.functional as F
            from sklearn.base import BaseEstimator, RegressorMixin
            from torch.utils.data import DataLoader, TensorDataset
            import torch.optim as optim
            
            class PyTorchRegressor(BaseEstimator, RegressorMixin):
                def __init__(self, hidden_sizes=(100, 50), learning_rate=0.001,
                             epochs=100, batch_size=32, device='mps'):
                    self.hidden_sizes = hidden_sizes
                    self.learning_rate = learning_rate
                    self.epochs = epochs
                    self.batch_size = batch_size
                    self.device = device
                    self.model = None
                    self.scaler = None
                
                def _create_model(self, input_size):
                    layers = []
                    prev_size = input_size
                    
                    for hidden_size in self.hidden_sizes:
                        layers.append(nn.Linear(prev_size, hidden_size))
                        layers.append(nn.ReLU())
                        layers.append(nn.Dropout(0.2))
                        prev_size = hidden_size
                    
                    layers.append(nn.Linear(prev_size, 1))
                    
                    return nn.Sequential(*layers)
                
                def fit(self, X, y):
                    import torch
                    from sklearn.preprocessing import StandardScaler
                    
                    # Set device
                    if (self.device == 'mps' and
                            torch.backends.mps.is_available()):
                        device = torch.device('mps')
                    else:
                        device = torch.device('cpu')
                    
                    # Scale features
                    self.scaler = StandardScaler()
                    X_scaled = self.scaler.fit_transform(X)
                    
                    # Convert to tensors
                    X_tensor = torch.FloatTensor(X_scaled).to(device)
                    y_tensor = torch.FloatTensor(y.reshape(-1, 1)).to(device)
                    
                    # Create model
                    self.model = self._create_model(X.shape[1]).to(device)
                    optimizer = optim.Adam(
                        self.model.parameters(), lr=self.learning_rate
                    )
                    criterion = nn.MSELoss()
                    
                    # Create data loader
                    dataset = TensorDataset(X_tensor, y_tensor)
                    dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
                    
                    # Training loop
                    self.model.train()
                    for epoch in range(self.epochs):
                        for batch_X, batch_y in dataloader:
                            optimizer.zero_grad()
                            outputs = self.model(batch_X)
                            loss = criterion(outputs, batch_y)
                            loss.backward()
                            optimizer.step()
                    
                    return self
                
                def predict(self, X):
                    import torch
                    
                    # Set device
                    if (self.device == 'mps' and
                            torch.backends.mps.is_available()):
                        device = torch.device('mps')
                    else:
                        device = torch.device('cpu')
                    
                    # Scale features
                    X_scaled = self.scaler.transform(X)
                    X_tensor = torch.FloatTensor(X_scaled).to(device)
                    
                    # Predict
                    self.model.eval()
                    with torch.no_grad():
                        predictions = self.model(X_tensor)
                    
                    return predictions.cpu().numpy().flatten()
            
            # Add PyTorch models with different architectures
            models['PyTorchMLP_Small'] = (PyTorchRegressor(hidden_sizes=(50,), device='mps'), {
                'learning_rate': [0.001, 0.01, 0.1],
                'epochs': [50, 100, 200],
                'batch_size': [16, 32, 64]
            })
            
            models['PyTorchMLP_Medium'] = (PyTorchRegressor(hidden_sizes=(100, 50), device='mps'), {
                'learning_rate': [0.001, 0.01, 0.1],
                'epochs': [50, 100, 200],
                'batch_size': [16, 32, 64]
            })
            
            models['PyTorchMLP_Large'] = (PyTorchRegressor(hidden_sizes=(200, 100, 50), device='mps'), {
                'learning_rate': [0.001, 0.01, 0.1],
                'epochs': [50, 100, 200],
                'batch_size': [16, 32, 64]
            })
            
            print("Added PyTorch models with Apple ML accelerator support")
            
        except ImportError as e:
            print(f"Could not add PyTorch models: {e}")
    
    # Add Advanced Ensemble Methods (after all base models are defined)
    try:
        # Voting Regressor - combines multiple models
        base_models = [
            ('rf', RandomForestRegressor(n_estimators=50, random_state=42)),
            ('gb', GradientBoostingRegressor(n_estimators=50, random_state=42)),
            ('ridge', Ridge(alpha=1.0))
        ]
        
        models['VotingRegressor'] = (VotingRegressor(estimators=base_models), {
            'weights': [[1, 1, 1], [2, 1, 1], [1, 2, 1], [1, 1, 2]]
        })
        
        # Stacking Regressor - meta-learning approach
        models['StackingRegressor'] = (StackingRegressor(
            estimators=[
                ('rf', RandomForestRegressor(n_estimators=50, random_state=42)),
                ('gb', GradientBoostingRegressor(n_estimators=50, random_state=42))
            ],
            final_estimator=Ridge(alpha=1.0),
            cv=3
        ), {
            'final_estimator__alpha': [0.1, 1.0, 10.0]
        })
        
        print("Added advanced ensemble methods (VotingRegressor, StackingRegressor)")
        
    except Exception as e:
        print(f"Could not add advanced ensemble methods: {e}")
    
    # Add more specialized regression models for comprehensive evaluation
    try:
        # Additional SVR variations with different kernels
        models['SVR_RBF'] = (SVR(kernel='rbf'), {
            'C': [0.1, 1, 10, 100, 1000],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
            'epsilon': [0.01, 0.1, 0.2, 0.5]
        })
        
        models['SVR_Poly'] = (SVR(kernel='poly'), {
            'C': [0.1, 1, 10, 100],
            'degree': [2, 3, 4, 5],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
            'epsilon': [0.01, 0.1, 0.2, 0.5]
        })
        
        models['SVR_Sigmoid'] = (SVR(kernel='sigmoid'), {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
            'epsilon': [0.01, 0.1, 0.2, 0.5]
        })
        
        # Additional KNN variations
        models['KNeighborsRegressor_Uniform'] = (KNeighborsRegressor(weights='uniform'), {
            'n_neighbors': [3, 5, 7, 10, 15, 20, 25],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'leaf_size': [10, 20, 30, 50],
            'p': [1, 2]  # Manhattan vs Euclidean distance
        })
        
        models['KNeighborsRegressor_Distance'] = (KNeighborsRegressor(weights='distance'), {
            'n_neighbors': [3, 5, 7, 10, 15, 20, 25],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'leaf_size': [10, 20, 30, 50],
            'p': [1, 2]  # Manhattan vs Euclidean distance
        })
        
        # Additional decision tree variations
        models['DecisionTreeRegressor_Deep'] = (DecisionTreeRegressor(random_state=42), {
            'max_depth': [5, 10, 20, None],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 4, 8],
            'max_features': ['auto', 'sqrt', 'log2', None],
            'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
        })
        
        models['ExtraTreeRegressor_Deep'] = (ExtraTreeRegressor(random_state=42), {
            'max_depth': [5, 10, 20, None],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 4, 8],
            'max_features': ['auto', 'sqrt', 'log2', None],
            'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
        })
        
        # Additional ensemble variations with more estimators
        models['RandomForestRegressor_Large'] = (RandomForestRegressor(random_state=42), {
            'n_estimators': [200, 300, 500],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False]
        })
        
        models['ExtraTreesRegressor_Large'] = (ExtraTreesRegressor(random_state=42), {
            'n_estimators': [200, 300, 500],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False]
        })
        
        # Additional neural network variations
        models['MLPRegressor_Deep'] = (MLPRegressor(random_state=42, max_iter=1000), {
            'hidden_layer_sizes': [(200, 100, 50), (300, 200, 100), (500, 300, 100), (1000, 500, 100)],
            'activation': ['relu', 'tanh', 'logistic'],
            'alpha': [0.0001, 0.001, 0.01, 0.1],
            'learning_rate_init': [0.001, 0.01, 0.1],
            'solver': ['adam', 'lbfgs', 'sgd'],
            'batch_size': ['auto', 32, 64, 128]
        })
        
        models['MLPRegressor_Wide'] = (MLPRegressor(random_state=42, max_iter=1000), {
            'hidden_layer_sizes': [(500,), (1000,), (2000,), (500, 500), (1000, 1000)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate_init': [0.001, 0.01, 0.1],
            'solver': ['adam', 'lbfgs']
        })
        
        # Additional boosting variations
        models['GradientBoostingRegressor_Deep'] = (GradientBoostingRegressor(random_state=42), {
            'n_estimators': [200, 300, 500],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7, 10, 15],
            'subsample': [0.8, 0.9, 1.0],
            'max_features': ['sqrt', 'log2', None],
            'criterion': ['squared_error', 'friedman_mse', 'absolute_error']
        })
        
        models['AdaBoostRegressor_Deep'] = (AdaBoostRegressor(random_state=42), {
            'n_estimators': [100, 200, 300, 500],
            'learning_rate': [0.01, 0.05, 0.1, 0.5, 1.0],
            'loss': ['linear', 'square', 'exponential']
        })
        
        # Additional bagging variations
        models['BaggingRegressor_Large'] = (BaggingRegressor(random_state=42), {
            'n_estimators': [50, 100, 200],
            'max_samples': [0.5, 0.7, 0.9, 1.0],
            'max_features': [0.5, 0.7, 0.9, 1.0],
            'bootstrap': [True, False],
            'bootstrap_features': [True, False]
        })
        
        # Additional specialized models
        models['QuantileRegressor'] = (QuantileRegressor(quantile=0.5, alpha=1.0), {
            'quantile': [0.1, 0.25, 0.5, 0.75, 0.9],
            'alpha': [0.1, 1.0, 10.0, 100.0]
        })
        
        models['TheilSenRegressor'] = (TheilSenRegressor(random_state=42), {
            'max_subpopulation': [1000, 5000, 10000, 20000],
            'n_subsamples': [None, 100, 500, 1000],
            'max_iter': [100, 300, 500]
        })
        
        models['RANSACRegressor'] = (RANSACRegressor(random_state=42), {
            'min_samples': [0.5, 0.7, 0.9],
            'residual_threshold': [None, 1.0, 2.0, 5.0],
            'max_trials': [50, 100, 200, 500]
        })
        
        models['KernelRidge'] = (KernelRidge(alpha=1.0), {
            'alpha': [0.1, 1.0, 10.0, 100.0],
            'kernel': ['linear', 'rbf', 'polynomial', 'sigmoid'],
            'gamma': [0.1, 1.0, 10.0, 100.0],
            'degree': [2, 3, 4, 5]
        })
        
        print("Added comprehensive set of specialized regression models")
        
    except Exception as e:
        print(f"Could not add specialized models: {e}")
    
    return models

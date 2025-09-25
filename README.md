# PerfCount: Cross-Platform Performance Prediction System

A comprehensive machine learning pipeline for predicting performance across different hardware platforms using performance counter data.

## Features

- **30+ Regression Models**: Comprehensive suite of scikit-learn models plus XGBoost, CatBoost, LightGBM
- **GPU Acceleration**: Apple Silicon MPS support for PyTorch-based neural networks (on Apple Silicon Macs)
- **Advanced Hyperparameter Tuning**: Grid, Random, and Halving search strategies
- **Robust Evaluation**: Leave-one-out cross-validation with statistical significance testing
- **Feature Analysis**: Importance analysis and visualization
- **Results Persistence**: Save models, results, and visualizations
- **Parallel Processing**: Multi-core training and evaluation

## Requirements

- Python 3.8+
- macOS (for Apple ML accelerator support), Linux, or Windows
- 4GB+ RAM recommended
- 2GB+ disk space

## Installation

### Quick Install

```bash
# Clone the repository
git clone <repository-url>
cd PerfCount

# Run the installation script
./install.sh
```

### Manual Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install core dependencies
pip install -r requirements-core.txt

# Install advanced features (optional)
pip install -r requirements-advanced.txt

# Install development tools (optional)
pip install -r requirements-dev.txt
```

### Using pip

```bash
# Install core version
pip install -e .

# Install with advanced features
pip install -e .[advanced]

# Install with development tools
pip install -e .[dev]
```

## Usage

### Quick Start

```bash
# Setup environment
./setup_venv.sh
source activate_venv.sh

# Run cross-platform prediction
python scripts/evaluation/cross_platform_prediction/cross_platform.py \
    --source_dir data/arm_server/3.0GHz/ \
    --target_dir data/arm_server/1.5GHz/ \
    --target_column "execution_time"
```

### Performance Modes

**Full Mode (Default)**: Trains 50+ models with comprehensive hyperparameter search
```bash
python scripts/evaluation/cross_platform_prediction/cross_platform.py \
    --source_dir source/ --target_dir target/ --target_column metric
```

**Fast Mode**: Reduced search space for quick exploration
```bash
python scripts/evaluation/cross_platform_prediction/cross_platform.py \
    --source_dir source/ --target_dir target/ --target_column metric \
    --fast_mode
```

**Limited Mode**: Train only specified number of models
```bash
python scripts/evaluation/cross_platform_prediction/cross_platform.py \
    --source_dir source/ --target_dir target/ --target_column metric \
    --max_models 10
```

### Comprehensive Documentation

For detailed usage instructions, command-line options, and advanced examples, see:
**[Cross-Platform Prediction README](scripts/evaluation/cross_platform_prediction/README.md)**

## 📁 Project Structure

```
PerfCount/
├── scripts/
│   ├── data_collection/          # Data collection scripts
│   └── evaluation/
│       └── cross_platform_prediction/
│           ├── cross_platform.py      # Main entry point
│           ├── test_gpu.py            # GPU testing script
│           └── modules/               # Modular components
│               ├── data_processing.py # Data loading and preprocessing
│               ├── model_config.py    # Model configurations
│               ├── model_training.py  # Training and hyperparameter tuning
│               ├── evaluation.py      # Evaluation metrics and analysis
│               ├── visualization.py   # Plotting and visualization
│               └── utils.py           # Utility functions
├── requirements-core.txt         # Core dependencies
├── requirements-advanced.txt     # Advanced features
├── requirements-dev.txt          # Development tools
├── setup.py                      # Package setup
└── install.sh                    # Installation script
```

## Configuration

### Model Selection

The system includes 30+ regression models:

- **Linear Models**: LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge, HuberRegressor, SGDRegressor, PassiveAggressiveRegressor, QuantileRegressor, TheilSenRegressor, RANSACRegressor
- **Tree Models**: DecisionTreeRegressor, ExtraTreeRegressor, RandomForestRegressor, ExtraTreesRegressor
- **Ensemble Methods**: GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor, HistGradientBoostingRegressor, VotingRegressor, StackingRegressor
- **Neural Networks**: MLPRegressor, PyTorchMLP (with GPU support)
- **Kernel Methods**: SVR, LinearSVR, KernelRidge
- **Advanced Boosting**: XGBoost, CatBoost, LightGBM
- **Specialized**: GaussianProcessRegressor, IsotonicRegression

### Hyperparameter Search Strategies

- **Grid Search**: Exhaustive search over parameter grid
- **Random Search**: Random sampling from parameter space
- **Halving Grid Search**: Successive halving with grid search
- **Halving Random Search**: Successive halving with random search
- **Auto**: Automatically selects best strategy based on data size

## Output

The system generates comprehensive outputs:

- **Results JSON**: Detailed performance metrics for all models
- **Model Rankings**: CSV files with model performance rankings
- **Visualizations**: Performance plots, feature importance, prediction vs actual
- **Trained Models**: Pickle files of best-performing models (optional)
- **Statistical Tests**: Significance testing results
- **Confidence Intervals**: Prediction uncertainty estimates

## Testing

```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=scripts tests/

# Test GPU functionality
python scripts/evaluation/cross_platform_prediction/test_gpu.py
```

## Development

### Code Formatting

```bash
# Format code
black scripts/
isort scripts/

# Lint code
flake8 scripts/
```

### Documentation

```bash
# Build documentation
cd docs/
make html
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- scikit-learn team for the excellent ML library
- XGBoost, CatBoost, and LightGBM teams for advanced boosting
- PyTorch team for deep learning and Apple MPS support
- The open-source community for inspiration and tools

## 📞 Support

For questions, issues, or contributions:

- Create an issue on GitHub
- Check the documentation
- Review the test examples

---

**Happy predicting! 🚀**

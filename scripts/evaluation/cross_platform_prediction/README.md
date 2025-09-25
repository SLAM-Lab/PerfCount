# Cross-Platform Performance Prediction

A comprehensive machine learning pipeline for predicting performance across different hardware platforms using performance counter data. This system performs leave-one-out cross-validation across multiple workloads, training 50+ regression models with advanced hyperparameter tuning.

## Features

- **50+ Regression Models**: Comprehensive suite including linear, tree-based, ensemble, neural network, and specialized models
- **Advanced Hyperparameter Tuning**: Grid, Random, and Halving search strategies with intelligent parameter space reduction
- **GPU Acceleration**: Apple Silicon MPS support for PyTorch-based neural networks (on Apple Silicon Macs)
- **Maximum Parallelism**: Utilizes all available CPU cores for parallel model training
- **Robust Evaluation**: Leave-one-out cross-validation with statistical significance testing
- **Feature Analysis**: Importance analysis and comprehensive visualizations
- **Results Persistence**: Save models, results, and visualizations in multiple formats
- **Performance Optimization**: Fast mode and model limiting for quick experimentation

## Quick Start

### 1. Setup Environment

```bash
# Create and activate virtual environment with all dependencies
./setup_venv.sh
source activate_venv.sh

# Or manually activate
source venv/bin/activate
```

### 2. Basic Usage

```bash
# Run cross-platform prediction with default settings
python cross_platform.py \
    --source_dir /path/to/source/csvs \
    --target_dir /path/to/target/csvs \
    --target_column "target_metric_name"
```

### 3. Example with Real Data

```bash
# Example with SPEC benchmark data
python cross_platform.py \
    --source_dir ../../../data/arm_server/3.0GHz/ \
    --target_dir ../../../data/arm_server/1.5GHz/ \
    --target_column "instructions_per_cycle" \
    --output_dir results/ \
    --experiment_name "spec_benchmark_analysis"
```

## Command Line Options

### Required Arguments
- `--source_dir`: Directory containing source CSV files
- `--target_dir`: Directory containing target CSV files  
- `--target_column`: Name of the target column to predict

### Hyperparameter Tuning
- `--cv_folds`: Number of CV folds for hyperparameter tuning (default: 5)
- `--search_strategy`: Search strategy - 'auto', 'grid', 'random', 'halving_grid', 'halving_random' (default: auto)
- `--max_iter`: Maximum iterations for random search (default: 50)

### Performance Optimization
- `--n_jobs`: Number of parallel jobs (default: all cores)
- `--use_gpu`: Use GPU acceleration when available (default: True)
- `--no_gpu`: Disable GPU acceleration (force CPU only)
- `--max_models`: Maximum number of models to train (for testing)
- `--fast_mode`: Reduce hyperparameter search space for speed

### Data Preprocessing
- `--no_preprocess`: Disable data preprocessing
- `--remove_outliers`: Remove outliers (default: True)
- `--feature_selection`: Number of features to select (default: None - use all)
- `--scaler_type`: Type of scaler - 'standard' or 'robust' (default: standard)

### Output and Analysis
- `--output_dir`: Output directory (default: results/)
- `--experiment_name`: Experiment name (default: auto-generated)
- `--save_models`: Save trained models to disk
- `--create_plots`: Create visualization plots (default: True)
- `--statistical_tests`: Perform statistical significance tests (default: True)
- `--feature_importance`: Analyze feature importance (default: True)

## Model Coverage

### Linear Models
- LinearRegression, Ridge, Lasso, ElasticNet
- BayesianRidge, HuberRegressor, SGDRegressor
- PassiveAggressiveRegressor, QuantileRegressor
- TheilSenRegressor, RANSACRegressor

### Tree-Based Models
- DecisionTreeRegressor, ExtraTreeRegressor
- RandomForestRegressor, ExtraTreesRegressor
- GradientBoostingRegressor, HistGradientBoostingRegressor
- AdaBoostRegressor, BaggingRegressor

### Support Vector Machines
- SVR (RBF, Polynomial, Sigmoid kernels)
- LinearSVR

### Nearest Neighbors
- KNeighborsRegressor (uniform and distance weights)
- RadiusNeighborsRegressor

### Neural Networks
- MLPRegressor (various architectures)
- PyTorchMLP (with Apple Silicon MPS support)
- GaussianProcessRegressor

### Ensemble Methods
- VotingRegressor, StackingRegressor
- XGBoost, CatBoost, LightGBM (if available)

### Specialized Models
- KernelRidge, IsotonicRegression

## Performance Modes

### Full Mode (Default)
- Trains all 50+ models with comprehensive hyperparameter search
- Uses all available CPU cores
- Best for final evaluation and publication

### Fast Mode
```bash
python cross_platform.py \
    --source_dir source/ --target_dir target/ --target_column metric \
    --fast_mode
```
- Reduces hyperparameter search space by 3x
- Maintains model diversity
- Good for initial exploration

### Limited Mode
```bash
python cross_platform.py \
    --source_dir source/ --target_dir target/ --target_column metric \
    --max_models 10
```
- Trains only specified number of models
- Selects diverse models across algorithm types
- Perfect for quick testing

## Output Structure

```
results/
├── experiment_20241201_143022/
│   ├── results.json              # Complete results
│   ├── results_summary.csv       # Summary table
│   ├── statistical_tests.json    # Significance tests
│   ├── models/                   # Saved models (if --save_models)
│   └── plots/                    # Visualizations
│       ├── model_performance.png
│       ├── feature_importance.png
│       └── prediction_vs_actual.png
```

## Data Format Requirements

### CSV Structure
- **Rows**: Performance counter samples
- **Columns**: Counter names + target column
- **Alignment**: Source and target CSVs must have matching workload names

### Example CSV
```csv
workload_name,instructions,cycles,cache_misses,target_metric
sample_1,1000000,500000,1000,2.0
sample_2,1000000,520000,1200,1.92
...
```

## Advanced Usage Examples

### High-Performance Run
```bash
python cross_platform.py \
    --source_dir source/ --target_dir target/ --target_column metric \
    --cv_folds 10 --search_strategy halving_random --max_iter 100 \
    --n_jobs -1 --save_models --create_plots
```

### Quick Experiment
```bash
python cross_platform.py \
    --source_dir source/ --target_dir target/ --target_column metric \
    --fast_mode --max_models 5 --cv_folds 3
```

### Custom Preprocessing
```bash
python cross_platform.py \
    --source_dir source/ --target_dir target/ --target_column metric \
    --remove_outliers --feature_selection 20 --scaler_type robust
```

## Testing and Validation

### Test GPU Acceleration (Apple Silicon)
```bash
python test_gpu.py
```

### Verify Installation
```bash
python -c "import modules.model_config; print('Installation successful')"
```

## Troubleshooting

### Common Issues

1. **Memory Issues**: Use `--max_models` to limit models or `--fast_mode` to reduce search space
2. **Slow Training**: Enable `--fast_mode` or reduce `--cv_folds`
3. **GPU Not Detected**: Check with `python test_gpu.py`
4. **Import Errors**: Ensure virtual environment is activated and dependencies installed

### Performance Tips

1. **Use Fast Mode**: For initial exploration, always start with `--fast_mode`
2. **Limit Models**: Use `--max_models 10` for quick testing
3. **Reduce CV Folds**: Use `--cv_folds 3` for faster hyperparameter tuning
4. **Parallel Processing**: Ensure `--n_jobs` is set to number of CPU cores

## Dependencies

### Core Requirements
- Python 3.8+
- pandas, numpy, scikit-learn
- matplotlib, seaborn

### Advanced Features
- XGBoost, CatBoost, LightGBM (optional)
- PyTorch (for Apple Silicon MPS support)
- statsmodels (for statistical tests)

### Development Tools
- pytest, black, flake8 (optional)

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{cross_platform_prediction,
  title={Cross-Platform Performance Prediction},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/PerfCount}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

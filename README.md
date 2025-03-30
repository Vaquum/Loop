# Loop

A project for working with cryptocurrency time series data.

## Project Architecture

The Loop project has been refactored to follow a more modular and maintainable architecture:

- `data.py`: Handles loading and splitting the raw data
  - Contains project-wide constants and configuration
  - Central source for feature column definitions and thresholds
- `features.py`: Performs feature engineering and scaling on raw data
  - Includes core feature extraction logic
  - Provides sequence preparation for transformer models
- `models/`: Directory containing model architecture code
  - `transformer.py`: Contains the model architecture for the transformer model
  - Includes PositionalEncoding class and model creation functions
- `utils/`: Directory containing utility functions
  - `callbacks.py`: Reusable training callbacks for TensorFlow/Keras
- `predict.py`: Handles loading models and making predictions
  - Provides confidence score output capabilities
  - Supports loading models by name or latest
- `backtest.py`: Runs backtests using models and features
  - Uses modular approach with smaller, focused methods
- `experiment.py`: Runs hyperparameter optimization and training
  - Includes utilities for model saving and management
- `saved_models/`: Root directory for storing trained models
  - Contains timestamped models and scalers
  - Primary location for model storage and retrieval

## Error Handling Philosophy

This project follows a strict "fail fast" philosophy:
- All functions use explicit error checking instead of silent fallbacks
- Critical errors raise appropriate exceptions instead of returning default values
- Hard validation is performed before operations to ensure all requirements are met

## Installation

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

## Usage

```python
from data import Data

# Create a data object
data_obj = Data()

# Access the data
df = data_obj.data
print(df.head())

# Handle NaN values (default is True)
data_obj = Data(drop_na=True)

# Split data into training, test, and validation sets (equal parts)
# This automatically prints split statistics
data_obj.split_data(1, 1, 1)

# Split without printing statistics
data_obj.split_data(1, 1, 1, print_stats=False)

# Access the split data
train_df = data_obj.input_data['train']
test_df = data_obj.input_data['test']
validate_df = data_obj.input_data['validate']

# Split with custom ratios (1:2:4)
data_obj.split_data(1, 2, 4)
# This gives:
# - 1/7 of the data for training
# - 2/7 of the data for testing
# - 4/7 of the data for validation

# Statistics output includes:
# - Row counts and percentages for each split
# - Date ranges for each split
```

## Transformer Model

The project includes a transformer-based model for market regime prediction using Features.production_features. The model can classify market conditions as bearish, neutral, or bullish based on sequences of technical indicators.

### Using the Transformer with Features

```python
from data import Data
from features import Features
from experiment_v2 import run_experiment_with_features

# Create and prepare data with Features
data_obj = Data()
data_obj.load_data('path/to/data.csv')
data_obj.split_data(1, 1, 1)

# Create features with StandardScaler (recommended for the transformer model)
features = Features(data_obj, scaler_type='standard')
features.process_features()  # Adds indicators and scales

# Run transformer model experiment
model, best_params = run_experiment_with_features(features, 'models/my_model.keras')
```

The transformer model specifically uses the following features for prediction:
- close_roc (price rate of change)
- volume, high, low, close (raw values)
- high_low_ratio, volume_change, high_close_ratio, low_close_ratio

The model creates sequences of these features (default: 30 days) and predicts the next day's market regime.

### Production Model Training

The experiment pipeline now includes an enhanced training process for production models:

1. **Hyperparameter Search**: Finds optimal parameters using 8:1:1 split (train:test:validate)

2. **Production Model Training**: 
   - Combines train and test data (9 parts) into a larger training set
   - Uses validation data (1 part) as the test set
   - Trains model with best parameters on this larger dataset
   - Provides better utilization of available data

3. **Versioned Model Storage**:
   - Saves models to 'production_models' directory with timestamps
   - Creates 'latest_model.keras' reference for easy access
   - Maintains history of production models

```python
from data import Data
import experiment_v2

# Default behavior now uses 8:1:1 split and trains production model
data_obj = Data()
model, scaler, best_params = experiment_v2.run_experiment()

# The production model is automatically saved to:
# - production_models/production_model_{timestamp}.keras
# - production_models/latest_model.keras (standard reference)
```

### Making Predictions with the Predict Class

The `Predict` class provides an easy way to load a trained transformer model and make predictions on new data:

```python
from data import Data
from features import Features
from predict import Predict

# Load the latest model from 'saved_models' directory
predictor = Predict()  # Always uses relative path 'saved_models' from root

# Predict latest market regime
latest = predictor.predict(data_obj.data)
print(f"Latest market regime: {predictor.get_market_regime_name(latest)}")

# Predict with features object
data_obj.split_data(8, 1, 1)
features = Features(data_obj, scaler_type='standard')
features.process_features()
regime = predictor.predict_with_features(features, 'validate')
```

The `Predict` class provides several methods:
- `predict()`: Makes predictions using raw data
- `predict_with_features()`: Makes predictions using Features.production_features
- `predict_and_explain()`: Returns detailed information about the prediction
- `get_market_regime_name()`: Converts a regime code to a human-readable name

Note: The `Predict` class uses a simple relative path approach, assuming the application is always run from the project root directory. It expects models to be in the `saved_models` directory with no fallbacks.

## Backtesting Trading Strategies

The project includes a robust backtesting system for evaluating day trading strategies based on market regime predictions.

### Using the BackTest Class

The `BackTest` class provides a scientifically sound approach to backtesting:

```python
from data import Data
from features import Features
from predict import Predict
from backtest import BackTest

# Setup data pipeline
data_obj = Data()
data_obj.split_data(8, 1, 1)
features = Features(data_obj, scaler_type='standard')
features.process_features()

# Create predictor (automatically loads latest model)
predictor = Predict()

# Create backtester
backtester = BackTest(
    predictor=predictor,
    features=features,
    initial_capital=10000.0,
    risk_per_trade=0.02,
    stop_loss=0.02,
    take_profit=0.04
)

# Run backtest
results = backtester.run_backtest()

# Generate report
metrics = backtester.generate_report(save_dir='backtest_reports')

# Print ROI (Return on Capital)
print(f"ROC: {metrics['roi']*100:.2f}%")

# Visualize results
backtester.plot_results()
```

### Backtest Parameters

The backtesting engine allows fine-tuning of several parameters:

- `initial_capital`: Starting capital for the simulation (default: $10,000)
- `risk_per_trade`: Percentage of capital to risk per trade (default: 2%)
- `stop_loss`: Stop loss percentage for risk management (default: 2%)
- `take_profit`: Take profit target percentage (default: 4%)
- `commission`: Trading commission percentage (default: 0.1%)
- `slippage`: Estimated slippage in execution (default: 0.1%)

### Trading Strategy Logic

The default strategy uses market regime predictions as follows:

1. Bullish prediction (2): Open or maintain long position
2. Bearish prediction (0): Open or maintain short position
3. Neutral prediction (1): Close any open positions, stay in cash

The backtester implements proper position sizing based on risk, incorporating:
- Stop-loss orders for all positions
- Take-profit targets
- Transaction costs (commissions and slippage)

### Performance Metrics

The backtester calculates comprehensive performance metrics:

- **Return on Capital (ROC)**: Primary performance metric
- **Sharpe Ratio**: Risk-adjusted return metric
- **Sortino Ratio**: Downside risk-adjusted return
- **Maximum Drawdown**: Worst peak-to-trough decline
- **Win Rate**: Percentage of winning trades
- **Profit Factor**: Gross profits divided by gross losses
- **Annual Return**: Annualized ROC

All metrics can be saved to CSV files and visualized through the reporting system.

### Backtesting

The `BackTest` class provides comprehensive backtesting capabilities for evaluating trading strategies:

- Scientifically robust day trading simulation
- Risk management with customizable parameters
- Detailed performance metrics and visualization
- Trade-by-trade analysis
- Interactive web-based dashboard for result visualization

Example:

```python
# Create backtester
backtester = BackTest(
    predictor=predictor,
    features=features,
    initial_capital=10000.0,
    risk_per_trade=0.02,
    stop_loss=0.02,
    take_profit=0.04
)

# Run backtest
results = backtester.run_backtest()

# Generate detailed report
metrics = backtester.generate_report(save_dir='backtest_reports')

# Launch interactive dashboard
app = backtester.dashboard(debug=True)
```

#### Dashboard Features

The dashboard provides an interactive web interface for analyzing backtest results:

- Key performance metrics (ROI, Sharpe ratio, drawdown, etc.)
- Interactive capital, drawdown, and position charts
- Prediction distribution and confidence analysis
- Detailed trade history table
- Color-coded performance indicators

To launch the dashboard:

```python
# Run the backtest first
results = backtester.run_backtest()

# Launch the dashboard on the default port (8050)
app = backtester.dashboard()

# Or specify a custom port and debug mode
app = backtester.dashboard(port=8051, debug=True)
```

## Technical Indicators

The `Features` class provides a workflow for adding technical indicators directly to your data and scaling it for machine learning.

```python
from data import Data
from features import Features

# Create and split data
data_obj = Data()
data_obj.split_data(1, 1, 1)

# Create features object (default uses RobustScaler)
features = Features(data_obj)

# Use StandardScaler instead (useful for transformer models)
features = Features(data_obj, scaler_type='standard')

# Option 1: Add indicators and scale in separate steps
features.add_indicators()  # Modifies features.input_data in place
features.scale_features()  # Creates features.production_features

# Option 2: Do everything in one step
features.process_features()  # Both adds indicators and scales

# Access the processed data
train_with_indicators = features.input_data['train']  # Original data with indicators added
train_production = features.production_features['train']  # Scaled data with indicators

# The data is now ready for machine learning!
X_train = train_production.drop(['open_time', 'close_time'], axis=1)
```

### Available Technical Indicators

When `add_indicators()` is called, the following indicators are added as new columns to each dataset (if the required data is available):

1. **RSI (Relative Strength Index)**
   - Added as `rsi_14` column

2. **Bollinger Bands**
   - Added as `bb_upper_20`, `bb_middle_20`, and `bb_lower_20` columns

3. **MACD (Moving Average Convergence Divergence)**
   - Added as `macd_line`, `macd_signal`, and `macd_histogram` columns

4. **EMAs (Exponential Moving Averages)**
   - Added as `ema_5`, `ema_10`, `ema_20`, etc. columns

5. **ATR (Average True Range)**
   - Added as `atr_14` column

6. **OBV (On-Balance Volume)**
   - Added as `obv` column

7. **ROC (Rate of Change)**
   - Uses existing `close_roc` column if available or adds `roc_12` column

8. **CCI (Commodity Channel Index)**
   - Added as `cci_20` column

9. **Ichimoku Cloud**
   - Added as `ichimoku_conversion_line`, `ichimoku_base_line`, `ichimoku_a`, and `ichimoku_b` columns

10. **Williams Fractals**
    - Added as `fractals_bearish_5` and `fractals_bullish_5` columns

11. **High/Low Ratio**
    - Added as `high_low_ratio` column

12. **Volume Change**
    - Added as `volume_change` column

13. **High/Close Ratio**
    - Added as `high_close_ratio` column

14. **Low/Close Ratio**
    - Added as `low_close_ratio` column

### Legacy Methods

For backward compatibility, you can still calculate individual indicators without adding them to the dataframes:

```python
rsi_values = features.rsi()  # Dictionary with RSI values for each dataset
bb_values = features.bollinger_bands(period=20, std_dev=2.0)  # Customize parameters
```

## Data Module Documentation

### Overview

The `data.py` module provides a robust interface for loading, processing, and splitting cryptocurrency time series data. It was designed with a focus on simplicity and reliability, following clean programming principles and the KISS (Keep It Simple, Stupid) philosophy.

### The `Data` Class

The `Data` class is the core component of this module, responsible for:

1. Loading cryptocurrency data from CSV files
2. Validating and converting data types
3. Handling missing values
4. Splitting data into training, testing, and validation sets

#### Initialization

```python
from data import Data

# Default initialization (drops NaN values)
data_obj = Data()

# Initialize without dropping NaN values
data_obj = Data(drop_na=False)
```

The `Data` class constructor accepts a single parameter:

- `drop_na` (bool, default=True): Controls whether rows with NaN values are dropped during initialization

#### Data Loading and Processing

When a `Data` object is instantiated:

1. The cryptocurrency data is loaded from the 'BTCUSDT-2020-2024.csv' file
2. If `drop_na=True`, rows containing NaN values are removed
3. The data types are validated and converted to appropriate formats:
   - Integer columns ('open_time', 'close_time', 'number_of_trades') are converted to int64
   - Float columns (price and volume data) are converted to float64

The loaded data can be accessed through the `data` attribute:

```python
# Access the full DataFrame
df = data_obj.data

# Get the first few rows
print(df.head())

# Check data types
print(df.dtypes)
```

#### Data Splitting Functionality

The `split_data` method provides a powerful way to divide the dataset into training, testing, and validation sets with customizable proportions:

```python
# Split data with equal ratios (1:1:1)
data_obj.split_data(train_ratio=1, test_ratio=1, validate_ratio=1)

# Split data with custom ratios (1:2:4)
data_obj.split_data(train_ratio=1, test_ratio=2, validate_ratio=4)

# Split without printing statistics
data_obj.split_data(train_ratio=1, test_ratio=1, validate_ratio=1, print_stats=False)
```

The method accepts the following parameters:

- `train_ratio` (int, default=1): Ratio for the training set
- `test_ratio` (int, default=1): Ratio for the testing set
- `validate_ratio` (int, default=1): Ratio for the validation set
- `print_stats` (bool, default=True): Whether to print splitting statistics

The data is split sequentially based on the given ratios. For example, with ratios 1:2:4:
- Training set: First 1/7 of the data (chronologically)
- Testing set: Next 2/7 of the data
- Validation set: Final 4/7 of the data

The split datasets are stored in the `input_data` dictionary:

```python
# Access individual splits
train_df = data_obj.input_data['train']
test_df = data_obj.input_data['test']
validate_df = data_obj.input_data['validate']
```

#### Data Validation

The `split_data` method includes robust validation to ensure reliable results:

1. Empty dataset validation: Prevents splitting an empty dataset
2. Input validation: Ensures all ratio parameters are non-negative integers
3. Minimum ratio validation: Requires at least one positive ratio
4. Row count validation: Verifies the dataset has enough rows for the requested split
5. Chronological validation: Ensures time series data is sorted by 'open_time'

If any validation fails, the method raises an informative `ValueError`.

#### Split Statistics

When `print_stats=True` (the default), the `split_data` method displays useful statistics about the split:

1. Total rows in the dataset
2. Rows and percentages for each split
3. Date ranges for each split (first and last date)

Example output:
```
=== Data Split Statistics ===
Total rows in dataset: 1820

Rows in each split:
  train: 260 rows (14.29%)
  test: 520 rows (28.57%)
  validate: 1040 rows (57.14%)

Date ranges for each split:
  train: 2020-01-02 to 2020-09-17
  test: 2020-09-18 to 2022-02-22
  validate: 2022-02-23 to 2024-12-30
===========================
```

### Use Cases

#### Basic Data Loading and Exploration

```python
from data import Data

# Load data
data_obj = Data()

# Explore data
print(f"Dataset shape: {data_obj.data.shape}")
print(f"Columns: {data_obj.data.columns.tolist()}")
print(f"Data types:\n{data_obj.data.dtypes}")
```

#### Training a Machine Learning Model

```python
from data import Data
import sklearn.linear_model as lm

# Load and split data
data_obj = Data()
data_obj.split_data(70, 15, 15)  # 70% train, 15% test, 15% validate

# Prepare features and target
X_train = data_obj.input_data['train'][['open', 'high', 'low', 'volume']]
y_train = data_obj.input_data['train']['close']

X_test = data_obj.input_data['test'][['open', 'high', 'low', 'volume']]
y_test = data_obj.input_data['test']['close']

# Train model
model = lm.LinearRegression()
model.fit(X_train, y_train)

# Evaluate
score = model.score(X_test, y_test)
print(f"Model R² score: {score:.4f}")
```

#### Time Series Analysis

```python
from data import Data
import matplotlib.pyplot as plt

# Load and split data
data_obj = Data()
data_obj.split_data(8, 1, 1)  # 80% historical, 20% recent

# Convert timestamps to datetime for better plotting
for split_name in data_obj.input_data:
    if 'open_time' in data_obj.input_data[split_name].columns:
        data_obj.input_data[split_name]['date'] = pd.to_datetime(
            data_obj.input_data[split_name]['open_time'], 
            unit='ms'
        )

# Plot close prices
plt.figure(figsize=(15, 7))
plt.plot(
    data_obj.input_data['train']['date'], 
    data_obj.input_data['train']['close'],
    label='Training Data'
)
plt.plot(
    data_obj.input_data['test']['date'], 
    data_obj.input_data['test']['close'],
    label='Testing Data', 
    color='orange'
)
plt.plot(
    data_obj.input_data['validate']['date'], 
    data_obj.input_data['validate']['close'],
    label='Validation Data', 
    color='green'
)
plt.title('Bitcoin Price (Split by Time Periods)')
plt.xlabel('Date')
plt.ylabel('Close Price (USD)')
plt.legend()
plt.tight_layout()
plt.show()
```

## Experiment Module

The Experiment module (`experiment.py`) allows you to:

- Run hyperparameter optimization for LSTM models
- Generate sequences from time series data
- Train models for predicting close prices
- Save optimized models for later use with the Predict module

To run an experiment:

```python
from loop.experiment import Experiment

# Create experiment with default parameters
experiment = Experiment()

# Run the experiment
results = experiment.run()
```

The experiment uses keras_tuner to find optimal hyperparameters for LSTM models in the context of algorithmic trading.

## Hyperparameter Optimization

The `search.py` module provides comprehensive hyperparameter tuning for LSTM models:

```python
from loop.search import run_search

# Run hyperparameter search with 10 trials
best_model, best_hyperparameters = run_search(
    data_path='BTCUSDT-2020-2024.csv',
    max_trials=10
)

# Print the best hyperparameters found
print(best_hyperparameters)
```

The module provides these capabilities:

- Hyperparameter optimization using Keras Tuner
- Hyperparameter ranges based on research for algorithmic trading
- Layer normalization between LSTM layers for improved stability
- Multiple optimization strategies (SGD, Adam, RMSprop)
- Comprehensive regularization options (L1, L2, L1L2)
- Automatic model and hyperparameter saving

Hyperparameters tuned include:
- LSTM units per layer
- Dropout rates
- Learning rates
- Activation functions
- Regularization types and values
- Batch sizes
- Optimizer selection

## Hyperparameter Search

Loop features a flexible hyperparameter search framework that can be used with any model type. The search framework is designed to be model-agnostic, allowing you to add new model types without changing the search code.

### Using the Search Framework

To use the search framework with your model, you need to implement the following interface:

```python
class YourModel:
    def prepare_data(self) -> Dict[str, np.ndarray]:
        """Return data splits for training and evaluation"""
        # Implementation...
        
    def get_tunable_model(self, hp: kt.HyperParameters, input_shape: Tuple) -> keras.Model:
        """Create and return a model with the given hyperparameters"""
        # Implementation...
        
    # Optional methods
    def get_callbacks(self) -> List[keras.callbacks.Callback]:
        """Return a list of callbacks for training"""
        # Implementation...
        
    def save_model_artifacts(self, model: keras.Model, hps: Dict[str, Any]) -> Tuple[str, ...]:
        """Save model and any model-specific artifacts"""
        # Implementation...
```

Then, you can run a search with:

```python
from loop.search import run_search
from loop.models.your_model import YourModel

best_model, best_hps = run_search(YourModel, max_trials=10)
```

### Built-in Models

Loop comes with several pre-configured model types:

- LSTM: `from loop.models.lstm import LSTM`

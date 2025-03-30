import xgboost as xgb
import numpy as np

def params():
    p = {'learning_rate': [0.008, 0.01, 0.015],  # Slightly wider range around current good values
         'max_depth': [3, 4, 5],                  # Allow slightly deeper trees
         'n_estimators': [150, 200, 250],         # More trees for better averaging
         'min_child_weight': [2, 3, 4],           # Better balance of detail vs noise
         'subsample': [0.8, 0.85, 0.9],          # Higher subsample for ROC prediction
         'colsample_bytree': [0.8, 0.85, 0.9],   # Higher feature sampling
         'gamma': [0.05, 0.1, 0.15],             # Moderate split conservatism
         'reg_alpha': [0.01, 0.05, 0.1],         # Light L1 regularization
         'reg_lambda': [0.5, 1.0, 2.0],          # Moderate L2 regularization
         'objective': ['reg:squarederror'],
         'booster': ['gbtree'],
         'early_stopping_rounds': [20, 25]}       # More patience for convergence
    
    return p


def prep(data, mode='train'):
    
    features = [
        # Price levels and gaps
        'open',         # Opening price for position entry
        'close',        # For calculating historical returns
        'high',         # Day's peak for momentum
        'low',          # Day's bottom for support

        # Volume indicators (trading pressure)
        'volume_vwap',  # Volume-weighted average price - key reference
        'volume_mfi',   # Money Flow Index - volume + price direction
        
        # Volatility (risk assessment)
        'volatility_atr',  # Average True Range - expected daily movement
        'volatility_bbw',  # Bollinger Band Width - volatility expansion/contraction
        
        # Enhanced trend detection
        'trend_adx',       # ADX strength - confirms if trend exists
        'trend_ema_fast',  # Fast EMA - recent price direction
        'trend_ema_slow',  # Slow EMA - longer-term direction
        'trend_macd',      # MACD - trend momentum and reversals
        
        # Momentum (near-term pressure)
        'momentum_rsi',    # RSI - classic overbought/oversold
        'momentum_ao',     # Awesome Oscillator - market momentum
        'momentum_stoch'   # Stochastic - price position within range
    ]
    
    if mode == 'train':
        labels = data[["close_roc"]].values
        return data[features].values, labels
    
    elif mode == 'predict':
        return data[features].values


def model(x_train, y_train, x_test, y_test, model_params):
    class DirectionalAccuracy:
        def __init__(self, threshold=0.5):
            self.threshold = threshold
            
        def __call__(self, y_true, y_pred):
            direction_true = np.diff(y_true, axis=0) >= 0
            direction_pred = np.diff(y_pred, axis=0) >= 0
            return np.mean(direction_true == direction_pred)

    class KerasStyleHistory:
        def __init__(self):
            self.history = {
                'loss': [],
                'val_loss': [],
                'mae': [],
                'val_mae': [],
                'rmse': [],
                'val_rmse': [],
                'dir_acc': [],
                'val_dir_acc': [],
                'mape': [],
                'val_mape': [],
                'feature_importance': None
            }

    class XGBModel:
        def __init__(self, model, history):
            self.model = model
            self.history = history
            
        def predict(self, x):
            return self.model.predict(x)
            
    model = xgb.XGBRegressor(
        learning_rate=model_params['learning_rate'],
        max_depth=model_params['max_depth'],
        n_estimators=model_params['n_estimators'],
        min_child_weight=model_params['min_child_weight'],
        subsample=model_params['subsample'],
        colsample_bytree=model_params['colsample_bytree'],
        gamma=model_params['gamma'],
        reg_alpha=model_params['reg_alpha'],
        reg_lambda=model_params['reg_lambda'],
        objective=model_params['objective'],
        booster=model_params['booster'],
        eval_metric=['mae', 'rmse'],
        early_stopping_rounds=model_params['early_stopping_rounds'],
        random_state=42
    )
    
    history = KerasStyleHistory()
    dir_acc_calculator = DirectionalAccuracy(threshold=0.5)
    
    eval_set = [(x_train, y_train), (x_test, y_test)]
    
    model.fit(
        x_train, y_train,
        eval_set=eval_set,
        verbose=False
    )
    
    # Calculate feature importance
    feature_names = [
        'open', 'close', 'high', 'low',
        'volume_vwap', 'volume_mfi',
        'volatility_atr', 'volatility_bbw',
        'trend_adx', 'trend_ema_fast', 'trend_ema_slow', 'trend_macd',
        'momentum_rsi', 'momentum_ao', 'momentum_stoch'
    ]
    importance_dict = {name: score for name, score in zip(feature_names, model.feature_importances_)}
    sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
    history.history['feature_importance'] = sorted_importance
    
    # Calculate metrics for history
    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)
    
    # Calculate metrics directly
    train_mae = np.mean(np.abs(y_train - train_pred))
    train_rmse = np.sqrt(np.mean((y_train - train_pred) ** 2))
    test_mae = np.mean(np.abs(y_test - test_pred))
    test_rmse = np.sqrt(np.mean((y_test - test_pred) ** 2))
    
    # Training metrics
    history.history['loss'] = [train_rmse]
    history.history['mae'] = [train_mae]
    history.history['rmse'] = [train_rmse]
    history.history['dir_acc'] = [dir_acc_calculator(y_train, train_pred)]
    history.history['mape'] = [np.mean(np.abs((y_train - train_pred) / (y_train + 1e-8))) * 100]
    
    # Validation metrics
    history.history['val_loss'] = [test_rmse]
    history.history['val_mae'] = [test_mae]
    history.history['val_rmse'] = [test_rmse]
    history.history['val_dir_acc'] = [dir_acc_calculator(y_test, test_pred)]
    history.history['val_mape'] = [np.mean(np.abs((y_test - test_pred) / (y_test + 1e-8))) * 100]
    
    # Create wrapper to match Keras interface
    wrapped_model = XGBModel(model, history)
    
    return wrapped_model, history

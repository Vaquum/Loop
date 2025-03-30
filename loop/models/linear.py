'''
Here is an example of a single-file model which has particular outputs 
to make it work with TradingLoop. 

The model must have the following functions:

- params(): Returns a dictionary of model parameters
- prep(): Prepares the data for training or prediction
- model(): Defines the model

The outputs of the functions must be in the following format:

- params(): Returns a dictionary of model parameters
- prep(): Returns a tuple of (X_train, y_train, X_test, y_test)
- model(): Returns a trained model

'''

import tensorflow as tf
from loop.utils.metrics import DirectionalAccuracy
from loop.utils.losses import ATRLoss


def params():
    p = {'learning_rate': [1e-6, 5e-6, 1e-5],  # Even lower learning rates
         'batch_size': [32, 64, 128],  # Batch size options
         'epsilon': [1e-7, 1e-6],  # Layer normalization stability
         'beta_1': [0.9, 0.95],  # Momentum range
         'beta_2': [0.999],  # Adam second moment
         'l1_reg': [0.0, 0.0001],  # Very light L1 regularization
         'l2_reg': [0.0, 0.0001],  # Very light L2 regularization
         'delta': [1.0],  # Huber loss delta
         'dir_threshold': [0.5],  # DirectionalAccuracy threshold
         'clip_norm': [1.0],  # Gradient clipping
         'relative_weight': [0.01]}  # Weight for relative error term
    
    return p


def prep(data, mode='train'):
    
    features = [
        # Price and Volume Fundamentals
        'open', 'high', 'low', 'close',
        'volume', 'number_of_trades', 
        'taker_buy_base_volume', 'taker_buy_quote_volume',
        
        # Volume Indicators
        'volume_obv', 'volume_vwap', 'volume_mfi', 'volume_cmf',
        
        # Volatility Indicators
        'volatility_atr', 'volatility_bbw', 'volatility_bbp',
        
        # Trend Indicators
        'trend_macd', 'trend_macd_diff',
        'trend_adx', 'trend_adx_pos', 'trend_adx_neg',
        'trend_cci', 'trend_aroon_up', 'trend_aroon_down',
        
        # Momentum Indicators
        'momentum_rsi', 'momentum_stoch',
        'momentum_ao', 'momentum_ppo',
        'momentum_kama']
    
    if mode == 'train':
        # Get labels and normalize them
        labels = data[["close_roc"]].values
        labels_mean = labels.mean()
        labels_std = labels.std()
        normalized_labels = (labels - labels_mean) / (labels_std + tf.keras.backend.epsilon())
        
        return data[features].values, normalized_labels
    
    elif mode == 'predict':
        return data[features].values


def model(x_train, y_train, x_test, y_test, model_params):
    # Create regularizer combining L1 and L2
    regularizer = tf.keras.regularizers.l1_l2(
        l1=model_params.get('l1_reg', 0.0001),
        l2=model_params.get('l2_reg', 0.0001)
    )

    # Linear regression model with non-negative output
    model = tf.keras.Sequential([
        tf.keras.layers.Input((x_train.shape[1], )),
        tf.keras.layers.LayerNormalization(epsilon=model_params['epsilon'], center=True, scale=True),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(1, 
                            kernel_regularizer=regularizer,
                            kernel_initializer='he_normal'),
        tf.keras.layers.ReLU()  # Separate ReLU layer for clarity
    ])

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=model_params['learning_rate'],
        beta_1=model_params['beta_1'],
        beta_2=model_params['beta_2'],
        clipnorm=model_params.get('clip_norm', 1.0)  # Add gradient clipping
    )
    
    # Create loss function instance
    loss_fn = ATRLoss(
        delta=model_params['delta'],
        relative_weight=model_params.get('relative_weight', 0.01)
    )
    
    model.compile(optimizer=optimizer,
                 loss=loss_fn,
                 metrics=[
                     tf.keras.metrics.MeanAbsoluteError(name='mae'),
                     tf.keras.metrics.MeanAbsolutePercentageError(name='mape'),
                     tf.keras.metrics.RootMeanSquaredError(name='rmse'),
                     DirectionalAccuracy(name='dir_acc', threshold=model_params['dir_threshold'])
                 ])

    # Define callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_mae',
            patience=15,
            restore_best_weights=True,
            min_delta=1e-4,
            mode='min',
            verbose=0
        ),
        
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.7,
            patience=3,
            min_lr=1e-6,
            min_delta=1e-4,
            mode='min',
            verbose=0
        ),
        
        # Add training progress monitoring
        tf.keras.callbacks.CSVLogger('training_log.csv', append=True),
    ]

    history = model.fit(x_train,
                       y_train,
                       validation_data=[x_test, y_test],
                       epochs=100,
                       callbacks=callbacks,
                       batch_size=model_params['batch_size'],
                       verbose=0)
    
    return model, history

import tensorflow as tf
from tensorflow import keras

# Check if tqdm is available
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

def create_callbacks(patience_early_stop=10, patience_reduce_lr=5, min_lr=1e-6):
    """
    Create standard callbacks for model training.
    
    Parameters:
        patience_early_stop: Patience for early stopping (default: 10)
        patience_reduce_lr: Patience for learning rate reduction (default: 5)
        min_lr: Minimum learning rate (default: 1e-6)
        
    Returns:
        List of callbacks
    """
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience_early_stop,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=patience_reduce_lr,
            min_lr=min_lr
        )
    ]
    
    # Add tqdm callback if available
    if TQDM_AVAILABLE:
        callbacks.append(create_tqdm_callback())
    
    return callbacks

def create_tqdm_callback():
    """
    Create TQDM progress bar callback.
    
    Returns:
        TqdmCallback instance
    """
    class TqdmCallback(keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            self.tqdm_progress = tqdm(
                total=len(self.model.history.epoch) + 1, 
                desc="Training", 
                initial=epoch
            )
            
        def on_epoch_end(self, epoch, logs=None):
            self.tqdm_progress.update(1)
            self.tqdm_progress.set_postfix({k: f"{v:.4f}" for k, v in logs.items()})
            
        def on_train_end(self, logs=None):
            self.tqdm_progress.close()
    
    return TqdmCallback() 
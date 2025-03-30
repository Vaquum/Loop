from tensorflow import keras
import tensorflow as tf


@keras.saving.register_keras_serializable(package='loop.utils')
class ATRLoss(tf.keras.losses.Loss):
    '''Custom loss function combining Huber loss with relative error for ATR prediction.
    
    Args:
        delta (float): The point where the Huber loss function changes from quadratic to linear.
        relative_weight (float): Weight for the relative error component.
        name (str): Name of the loss function.
        reduction (str): Type of reduction to apply to the loss.
    '''
    def __init__(
        self, 
        delta: float = 1.0, 
        relative_weight: float = 0.01, 
        name: str = 'atr_loss',
        reduction: str = 'mean'
    ):
        super().__init__(reduction=reduction, name=name)
        self.delta = delta
        self.relative_weight = relative_weight
        self.huber = tf.keras.losses.Huber(
            delta=delta, 
            reduction=tf.keras.losses.Reduction.NONE
        )

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        # Ensure positive predictions
        y_pred = tf.maximum(y_pred, tf.keras.backend.epsilon())
        
        # Huber loss with proper reduction
        huber = tf.reduce_mean(self.huber(y_true, y_pred))
        
        # Relative error with proper reduction
        rel_error = tf.reduce_mean(
            tf.abs(y_true - y_pred) / (tf.abs(y_true) + tf.keras.backend.epsilon())
        )
        
        # Combine losses with reduced relative error weight
        return huber + self.relative_weight * rel_error

    def get_config(self) -> dict:
        base_config = super().get_config()
        return {
            **base_config,
            'delta': self.delta,
            'relative_weight': self.relative_weight
        }
import tensorflow as tf
from tensorflow.keras.metrics import Metric
from tensorflow.keras.saving import register_keras_serializable


@register_keras_serializable(package='loop.utils.metrics')
class DirectionalAccuracy(Metric):
    '''Measures accuracy of predicted direction changes in time series.
    
    This metric is particularly useful for financial time series where predicting
    the direction of change (up/down) is as important as the magnitude.
    
    Attributes:
        correct_predictions: Counter for correct directional predictions
        total_predictions: Counter for total predictions made
        threshold: Minimum change to consider as a direction change
    '''
    
    def __init__(self, name='directional_accuracy', threshold=0.0, **kwargs):
        super().__init__(name=name, **kwargs)
        self.correct_predictions = self.add_weight(
            name='correct_predictions',
            initializer='zeros'
        )
        self.total_predictions = self.add_weight(
            name='total_predictions',
            initializer='zeros'
        )
        self.threshold = threshold

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Calculate directions (1 for up, 0 for down)
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # Get changes
        true_changes = y_true[1:] - y_true[:-1]
        pred_changes = y_pred[1:] - y_pred[:-1]
        
        # Apply threshold to changes
        true_dirs = tf.cast(true_changes > self.threshold, tf.float32)
        pred_dirs = tf.cast(pred_changes > self.threshold, tf.float32)
        
        # Calculate correct predictions
        correct = tf.cast(true_dirs == pred_dirs, tf.float32)
        
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight[1:], tf.float32)
            correct = correct * sample_weight
        
        self.correct_predictions.assign_add(tf.reduce_sum(correct))
        self.total_predictions.assign_add(tf.cast(tf.size(correct), tf.float32))

    def result(self):
        return self.correct_predictions / (self.total_predictions + tf.keras.backend.epsilon())

    def reset_state(self):
        self.correct_predictions.assign(0.0)
        self.total_predictions.assign(0.0)

    def get_config(self):
        config = super().get_config()
        config.update({'threshold': self.threshold})
        return config 
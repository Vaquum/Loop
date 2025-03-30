import tensorflow as tf

class Predict:

    def __init__(self, predictor):
        
        self.predictor = predictor

    def predict(self, validate_data, prep):

        '''Give predictions for a given dataset'''

        predict_data = prep(validate_data, mode='predict')
            
        predictions = self.predictor(predict_data)
        probs = tf.constant(predictions)
        class_indices = tf.argmax(probs, axis=1) - 1
        
        return class_indices.numpy()
    
    def load_model(sef):

        return
    
    def save_model(self):

        return

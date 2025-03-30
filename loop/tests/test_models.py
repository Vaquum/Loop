"""
Test script for the transformer model in models/transformer.py.

This script tests the PositionalEncoding class and the
create_transformer_model function.
"""

import unittest
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import sys
import tempfile

# Add the parent directory to sys.path to import loop modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from loop.models.transformer import PositionalEncoding, create_transformer_model

class TestTransformerModel(unittest.TestCase):
    """Test class for transformer model components"""
    
    def test_positional_encoding(self):
        """Test the PositionalEncoding layer"""
        # Create a PositionalEncoding layer
        d_model = 64
        pos_encoding = PositionalEncoding(d_model)
        
        # Create input tensor
        batch_size = 2
        sequence_length = 30
        inputs = tf.random.uniform((batch_size, sequence_length, d_model))
        
        # Apply positional encoding
        outputs = pos_encoding(inputs)
        
        # Check output shape
        self.assertEqual(outputs.shape, (batch_size, sequence_length, d_model))
        
        # Check that outputs are different from inputs (encoding was applied)
        self.assertFalse(np.array_equal(inputs.numpy(), outputs.numpy()))
        
    def test_create_transformer_model(self):
        """Test the create_transformer_model function"""
        # Define parameters for the transformer model
        params = {
            'd_model': 64,
            'num_heads': 2,
            'ff_dim': 128,
            'num_transformer_blocks': 2,
            'mlp_units': [64, 32],
            'dropout_rate': 0.1,
            'mlp_dropout': 0.2,
            'learning_rate': 0.001,
            'batch_size': 32
        }
        
        # Create model
        model = create_transformer_model(params)
        
        # Check that model is a Keras model
        self.assertIsInstance(model, keras.Model)
        
        # Verify model input shape
        self.assertEqual(model.input_shape, (None, None, 10), "Unexpected input shape")
        
        # Verify model output shape
        self.assertEqual(model.output_shape, (None, 3), "Unexpected output shape")
        
        # Test model with sample input
        batch_size = 2
        sequence_length = 30
        n_features = 10
        
        # Create random input
        inputs = np.random.rand(batch_size, sequence_length, n_features)
        
        # Make prediction
        outputs = model.predict(inputs)
        
        # Check output shape
        self.assertEqual(outputs.shape, (batch_size, 3))
        
        # Check that outputs sum to approximately 1 (softmax)
        for i in range(batch_size):
            self.assertAlmostEqual(np.sum(outputs[i]), 1.0, places=5)

if __name__ == '__main__':
    unittest.main() 
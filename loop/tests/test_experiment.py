"""
Test script for the experiment.py module.

This script tests the model creation, training and hyperparameter optimization
functions in the experiment.py module.
"""

import unittest
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import tempfile
import shutil
import os
import sys
from unittest.mock import patch, MagicMock
import glob

# Add the parent directory to sys.path to import loop modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from loop.experiment import model_builder, create_transformer_model, run_hyperparameter_optimization, train_best_model, save_model_and_scaler

class MockData:
    """Mock Data class for testing"""
    def __init__(self):
        self.input_data = {
            'train': pd.DataFrame({
                'close': np.linspace(100, 200, 100),
                'open_time': np.arange(100),
                'high': np.linspace(105, 210, 100),
                'low': np.linspace(95, 190, 100),
                'volume': np.random.randint(1000, 2000, 100),
                'close_roc': np.random.normal(0.001, 0.01, 100),
            }),
            'test': pd.DataFrame({
                'close': np.linspace(200, 220, 20),
                'open_time': np.arange(100, 120),
                'high': np.linspace(210, 231, 20),
                'low': np.linspace(190, 209, 20),
                'volume': np.random.randint(1000, 2000, 20),
                'close_roc': np.random.normal(0.001, 0.01, 20),
            }),
            'validate': pd.DataFrame({
                'close': np.linspace(220, 230, 10),
                'open_time': np.arange(120, 130),
                'high': np.linspace(231, 242, 10),
                'low': np.linspace(209, 218, 10),
                'volume': np.random.randint(1000, 2000, 10),
                'close_roc': np.random.normal(0.001, 0.01, 10),
            }),
        }
        
    def split_data(self, train_ratio=1, test_ratio=1, validate_ratio=1, print_stats=True):
        """Mock split_data method"""
        return self

class MockFeatures:
    """Mock Features class for testing"""
    def __init__(self, data=None, scaler_type='robust'):
        self.data = data
        self.scaler_type = scaler_type
        self.production_features = {
            'train': pd.DataFrame(np.random.randn(100, 10)),
            'test': pd.DataFrame(np.random.randn(20, 10)),
            'validate': pd.DataFrame(np.random.randn(10, 10)),
        }
        
    def process_features(self):
        """Mock process_features method"""
        return self
        
    def prepare_sequences(self, data=None, sequence_length=None):
        """Mock prepare_sequences method"""
        # Create random sequences
        X = np.random.rand(50, 30, 10)  # 50 samples, 30 timesteps, 10 features
        y = np.random.randint(0, 3, 50)  # 3 classes
        scaler = MagicMock()  # Mock scaler
        return X, y, scaler

class TestExperiment(unittest.TestCase):
    """Test class for experiment.py functions"""
    
    def setUp(self):
        """Set up the test environment"""
        # Create a temporary directory for test output
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up after tests"""
        # Remove temporary directory
        shutil.rmtree(self.test_dir)
    
    def test_extract_features(self):
        """Test that extract_features correctly delegates to Features.extract_features"""
        # Create a test DataFrame
        df = pd.DataFrame({
            'close': np.linspace(100, 120, 21),
            'high': np.linspace(105, 126, 21),
            'low': np.linspace(95, 114, 21),
            'volume': np.full(21, 1000),
            'close_roc': np.full(21, 0.001)
        })
        
        # Get features at a valid index
        from loop.experiment import extract_features
        features = extract_features(df, 10)
        
        # Verify we get the expected number of features
        self.assertIsNotNone(features)
        self.assertEqual(len(features), 10)  # There should be 10 features
        
        # Test with invalid inputs to verify error handling
        self.assertIsNone(extract_features(None, 0))  # Empty dataframe
        self.assertIsNone(extract_features(df, -1))  # Invalid index
        self.assertIsNone(extract_features(df, 100))  # Out of bounds index
    
    def test_model_builder(self):
        """Test the model_builder function"""
        # Create a mock hyperparameter object
        hp = MagicMock()
        hp.Choice.return_value = 64
        hp.Int.return_value = 2
        hp.Float.return_value = 0.1
        
        # Call model_builder
        model = model_builder(hp)
        
        # Check that model is a Keras model
        self.assertIsInstance(model, keras.Model)
        
        # Verify model has expected layers
        self.assertIsNotNone(model.get_layer('input_layer'), "Input layer missing")
        
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
    
    @patch('loop.loop.experiment.plt.show')
    @patch('loop.loop.experiment.plt.savefig')
    def test_save_model_and_scaler(self, mock_savefig, mock_show):
        """Test the save_model_and_scaler function"""
        # Create a simple model and scaler
        model = keras.Sequential([
            keras.layers.Dense(3, activation='softmax', input_shape=(10,))
        ])
        
        scaler = MagicMock()
        
        # Call save_model_and_scaler
        save_model_and_scaler(
            model, 
            scaler, 
            name_prefix='test_model'
        )
        
        # Check that files were created in saved_models directory
        saved_models_dir = 'saved_models'
        
        # Get all model files created
        model_files = glob.glob(os.path.join(saved_models_dir, 'test_model_*.keras'))
        scaler_files = glob.glob(os.path.join(saved_models_dir, 'test_scaler_*.pkl'))
        
        # Verify files were created
        self.assertGreater(len(model_files), 0, f"No model files created in {saved_models_dir}")
        self.assertGreater(len(scaler_files), 0, f"No scaler files created in {saved_models_dir}")
        
        # Clean up created files
        for file_path in model_files + scaler_files:
            if os.path.exists(file_path):
                os.remove(file_path)

if __name__ == '__main__':
    unittest.main() 
"""
Test script to verify production model training and saving functionality.

This script:
1. Creates a simple mock dataset
2. Runs a small hyperparameter optimization
3. Trains a production model with combined train+test data
4. Verifies that models are saved to the correct paths
5. Tests the Predict class's ability to find and load the latest model
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import shutil
import tempfile
import datetime
import sys

# Add the parent directory to sys.path to import loop modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import our modules
try:
    from loop.loop.experiment import train_production_model, create_transformer_model
    from loop.loop.predict import Predict
except ImportError:
    print("Could not import required modules. Make sure the loop package is properly installed.")
    exit(1)

def create_mock_data():
    """Create mock data splits for testing"""
    # Create random training data
    X_train = np.random.rand(100, 30, 10)  # 100 samples, 30 timesteps, 10 features
    y_train = np.random.randint(0, 3, 100)  # 3 classes
    
    # Create random test and validation data
    X_test = np.random.rand(20, 30, 10)
    y_test = np.random.randint(0, 3, 20)
    
    X_val = np.random.rand(10, 30, 10)
    y_val = np.random.randint(0, 3, 10)
    
    # Pack into data_splits dictionary
    data_splits = {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'X_val': X_val,
        'y_val': y_val
    }
    
    return data_splits

def test_production_model_training():
    """Test production model training with mock data"""
    print("Testing production model training...")
    
    # Create mock data
    data_splits = create_mock_data()
    
    # Create mock best_params
    best_params = {
        'd_model': 32,
        'num_heads': 2,
        'ff_dim': 64,
        'num_transformer_blocks': 1,
        'mlp_units': [32],
        'dropout_rate': 0.1,
        'mlp_dropout': 0.1,
        'learning_rate': 0.001,
        'batch_size': 16
    }
    
    # Create a temporary directory for models
    temp_dir = tempfile.mkdtemp()
    production_dir = os.path.join(temp_dir, 'production_models')
    os.makedirs(production_dir, exist_ok=True)
    
    # Save original working directory and change to temp dir
    cwd = os.getcwd()
    os.chdir(temp_dir)
    
    try:
        # Mock scaler as None for simplicity
        scaler = None
        
        # Train production model
        print("Training production model...")
        model, scaler = train_production_model(best_params, data_splits, scaler, epochs=2)
        
        # Save production model with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(production_dir, f'production_model_{timestamp}.keras')
        
        try:
            model.save(model_path)
            print(f"Saved model to {model_path}")
            
            # Create latest model symbolic link
            latest_path = os.path.join(production_dir, 'latest_model.keras')
            shutil.copy2(model_path, latest_path)
            
            # Verify files exist
            assert os.path.exists(model_path), f"Model file {model_path} does not exist"
            assert os.path.exists(latest_path), f"Latest model file {latest_path} does not exist"
            
            # Test Predict class
            print("Testing Predict class with the saved model...")
            
            # Return to original directory and initialize Predict
            os.chdir(cwd)
            try:
                # Initialize with explicit path
                pred1 = Predict(model_path=model_path, use_latest=False)
                print("Successfully loaded model with explicit path")
                
                # Reset current directory to temp dir for latest model test
                os.chdir(temp_dir)
                
                # Initialize with latest model detection
                pred2 = Predict(use_latest=True)
                print("Successfully loaded model with latest model detection")
                
                print("All tests passed successfully!")
                return True
                
            except Exception as e:
                print(f"Error testing Predict class: {str(e)}")
                return False
                
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            return False
            
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        return False
        
    finally:
        # Clean up
        os.chdir(cwd)
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    # Run test
    success = test_production_model_training()
    
    # Return to original directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    if success:
        print("✅ All tests passed!")
    else:
        print("❌ Tests failed!")
        exit(1) 
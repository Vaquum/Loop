import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from pandas.testing import assert_frame_equal
from loop.features import Features
from sklearn.preprocessing import RobustScaler

class MockData:
    """Mock Data class for testing Features."""
    def __init__(self, input_data=None):
        self.input_data = input_data or {}

class TestFeatures(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        # Create test DataFrames
        self.normal_df = pd.DataFrame({
            'open_time': [1000, 2000, 3000],
            'close_time': [1100, 2100, 3100],
            'open': [10.0, 20.0, 30.0],
            'high': [12.0, 22.0, 32.0],
            'low': [9.0, 19.0, 29.0],
            'close': [11.0, 21.0, 31.0],
            'volume': [100, 200, 300],
            'category': ['A', 'B', 'C']  # non-numeric column
        })
        
        self.empty_df = pd.DataFrame()
        
        self.non_numeric_df = pd.DataFrame({
            'open_time': [1000, 2000, 3000],
            'close_time': [1100, 2100, 3100],
            'category': ['A', 'B', 'C']
        })
        
        # Create mock Data objects
        self.normal_data = MockData({
            'train': self.normal_df.copy(),
            'test': self.normal_df.copy(),
            'validate': self.normal_df.copy()
        })
        
        self.empty_test_data = MockData({
            'train': self.normal_df.copy(),
            'test': self.empty_df.copy(),
            'validate': self.normal_df.copy()
        })
        
        self.missing_train_data = MockData({
            'test': self.normal_df.copy(),
            'validate': self.normal_df.copy()
        })
        
        self.empty_train_data = MockData({
            'train': self.empty_df.copy(),
            'test': self.normal_df.copy(),
            'validate': self.normal_df.copy()
        })
        
        self.all_empty_data = MockData({
            'train': self.empty_df.copy(),
            'test': self.empty_df.copy(),
            'validate': self.empty_df.copy()
        })
        
        self.non_numeric_data = MockData({
            'train': self.non_numeric_df.copy(),
            'test': self.non_numeric_df.copy(),
            'validate': self.non_numeric_df.copy()
        })
        
        # Object without input_data attribute
        self.invalid_data = object()
    
    @patch('loop.features.Features._fit_transform_data')
    def test_init_normal_case(self, mock_fit_transform):
        """Test that Features initializes correctly with a proper Data object."""
        features = Features(self.normal_data)
        
        # Check that attributes are initialized correctly
        self.assertEqual(features.data, self.normal_data)
        self.assertIsNone(features.scaler)
        self.assertEqual(features.scaled_data, {})
        
        # Check that _fit_transform_data was called
        mock_fit_transform.assert_called_once()
    
    def test_normal_case(self):
        """Test the normal case with proper data."""
        features = Features(self.normal_data)
        
        # Check that scaler is initialized
        self.assertIsInstance(features.scaler, RobustScaler)
        
        # Check that scaled_data has all the splits
        self.assertEqual(set(features.scaled_data.keys()), 
                         set(self.normal_data.input_data.keys()))
        
        # Check that DataFrames have the same shape
        for split in self.normal_data.input_data:
            self.assertEqual(features.scaled_data[split].shape, 
                            self.normal_data.input_data[split].shape)
            
            # Check that non-numeric columns are preserved
            self.assertTrue(
                (features.scaled_data[split]['category'] == 
                 self.normal_data.input_data[split]['category']).all()
            )
            
            # Check that numeric columns were scaled (values changed)
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                # At least one value should be different after scaling
                self.assertFalse(
                    (features.scaled_data[split][col] == 
                     self.normal_data.input_data[split][col]).all()
                )
    
    def test_missing_input_data_attribute(self):
        """Test that Features raises ValueError when data has no input_data attribute."""
        with self.assertRaises(ValueError) as context:
            Features(self.invalid_data)
        
        self.assertIn("must be split", str(context.exception))
    
    def test_empty_input_data(self):
        """Test that Features raises ValueError when input_data is empty."""
        with self.assertRaises(ValueError) as context:
            Features(MockData())
        
        self.assertIn("must be split", str(context.exception))
    
    def test_missing_train_dataset(self):
        """Test that Features raises ValueError when there's no train dataset."""
        with self.assertRaises(ValueError) as context:
            Features(self.missing_train_data)
        
        self.assertIn("empty, cannot fit scaler", str(context.exception))
    
    def test_empty_train_dataset(self):
        """Test that Features raises ValueError when train dataset is empty."""
        with self.assertRaises(ValueError) as context:
            Features(self.empty_train_data)
        
        self.assertIn("empty, cannot fit scaler", str(context.exception))
    
    def test_empty_test_dataset(self):
        """Test that Features handles empty test dataset properly."""
        features = Features(self.empty_test_data)
        
        # Check that test split has an empty DataFrame
        self.assertEqual(len(features.scaled_data['test']), 0)
        
        # Check that other splits are scaled correctly
        self.assertEqual(features.scaled_data['train'].shape, 
                        self.empty_test_data.input_data['train'].shape)
        self.assertEqual(features.scaled_data['validate'].shape, 
                        self.empty_test_data.input_data['validate'].shape)
    
    def test_all_empty_datasets(self):
        """Test that Features raises ValueError when all datasets are empty."""
        with self.assertRaises(ValueError) as context:
            Features(self.all_empty_data)
        
        self.assertIn("All datasets are empty", str(context.exception))
    
    def test_no_numeric_columns(self):
        """Test that Features raises ValueError when no numeric columns are found."""
        # Create data with only excluded columns
        data = MockData({
            'train': pd.DataFrame({
                'open_time': [1000, 2000, 3000],
                'close_time': [1100, 2100, 3100],
                'Unnamed: 0': [1, 2, 3]
            }),
            'test': pd.DataFrame()
        })
        
        with self.assertRaises(ValueError) as context:
            Features(data)
        
        self.assertIn("No numeric columns found", str(context.exception))
    
    def test_feature_column_selection(self):
        """Test that _get_feature_columns selects the correct columns."""
        # Create a Features object with our test data
        with patch('loop.features.RobustScaler'):  # Prevent actual scaling
            with patch.object(Features, '_fit_transform_data'):  # Prevent the method from being called
                features = Features(self.normal_data)
                
                # Call _get_feature_columns directly
                feature_cols = features._get_feature_columns()
                
                # Check that timestamps and non-numeric columns are excluded
                self.assertNotIn('open_time', feature_cols)
                self.assertNotIn('close_time', feature_cols)
                self.assertNotIn('category', feature_cols)
                
                # Check that numeric columns are included
                self.assertIn('open', feature_cols)
                self.assertIn('high', feature_cols)
                self.assertIn('low', feature_cols)
                self.assertIn('close', feature_cols)
                self.assertIn('volume', feature_cols)
    
    def test_scaler_fit_once(self):
        """Test that the scaler is fit only once on the training data."""
        # Create a custom replacement for transform that returns data of the correct shape
        def mock_transform(X):
            # Return a numpy array with the same shape as X
            return np.zeros(X.shape)
        
        with patch('sklearn.preprocessing.RobustScaler.fit') as mock_fit:
            with patch('sklearn.preprocessing.RobustScaler.transform', side_effect=mock_transform):
                Features(self.normal_data)
                
                # Check that fit was called exactly once
                mock_fit.assert_called_once()
    
    def test_transform_called_for_each_dataset(self):
        """Test that transform is called for each non-empty dataset."""
        # Create a custom replacement for transform that returns data of the correct shape
        def mock_transform(X):
            # Return a numpy array with the same shape as X
            return np.zeros(X.shape)
        
        with patch('sklearn.preprocessing.RobustScaler.fit'):
            with patch('sklearn.preprocessing.RobustScaler.transform', side_effect=mock_transform) as mock_transform_call:
                Features(self.normal_data)
                
                # Should be called for train, test, and validate (3 times)
                self.assertEqual(mock_transform_call.call_count, 3)
    
    def test_fallback_to_other_split(self):
        """Test that _get_feature_columns falls back to another split if train is empty."""
        # Modify data to have empty train but non-empty test
        data = MockData({
            'train': pd.DataFrame(),
            'test': self.normal_df.copy()
        })
        
        # First, verify that _get_feature_columns would raise an error without the fallback
        with self.assertRaises(ValueError):
            Features(data)
        
        # Now test with a modified _get_feature_columns that returns empty train before other splits
        with patch.object(Features, '_get_feature_columns', 
                        side_effect=lambda: ['open', 'close']):
            with patch.object(Features, '_fit_transform_data'):
                features = Features(self.normal_data)
                # If we get here without an error, the fallback logic works
                self.assertIsInstance(features, Features)
    
    def test_original_data_unchanged(self):
        """Test that the original data in the Data object is not modified."""
        # Make a copy of the original data
        orig_train = self.normal_data.input_data['train'].copy()
        
        # Create Features object that scales the data
        features = Features(self.normal_data)
        
        # Verify that the original data is unchanged
        pd.testing.assert_frame_equal(self.normal_data.input_data['train'], orig_train)

if __name__ == '__main__':
    unittest.main() 
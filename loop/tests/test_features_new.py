import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from pandas.testing import assert_frame_equal
from loop.loop.features import Features

class MockData:
    """Mock Data class for testing Features."""
    def __init__(self, input_data=None):
        self.input_data = input_data or {}

class TestFeaturesNew(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        # Create test DataFrames with more rows to handle period requirements
        self.normal_df = pd.DataFrame({
            'open_time': list(range(1000, 1050)),
            'close_time': list(range(1100, 1150)),
            'open': [10.0 + i * 0.1 for i in range(50)],
            'high': [12.0 + i * 0.1 for i in range(50)],
            'low': [9.0 + i * 0.1 for i in range(50)],
            'close': [11.0 + i * 0.1 for i in range(50)],
            'volume': [100 + i * 10 for i in range(50)],
            'close_roc': [0.01 + i * 0.001 for i in range(50)]
        })
        
        self.empty_df = pd.DataFrame()
        
        # Create mock Data objects
        self.data = MockData({
            'train': self.normal_df.iloc[:30].copy(),
            'test': self.normal_df.iloc[30:40].copy(),
            'validate': self.normal_df.iloc[40:].copy()
        })
    
    def test_initialization(self):
        """Test that Features initializes correctly and absorbs input_data."""
        features = Features(self.data)
        
        # Check that input_data is absorbed correctly
        self.assertEqual(set(features.input_data.keys()), set(self.data.input_data.keys()))
        
        # Check that the data was copied, not referenced
        self.assertIsNot(features.input_data['train'], self.data.input_data['train'])
        
        # Verify that other attributes are initialized
        self.assertIsNone(features.scaler)
        self.assertEqual(features.production_features, {})
    
    def test_add_indicators(self):
        """Test that add_indicators adds indicator columns."""
        # Patch any print statements to avoid cluttering test output
        with patch('builtins.print'):
            features = Features(self.data)
            features.add_indicators()
            
            # Check that indicators are added to input_data
            train_df = features.input_data['train']
            
            # Indicators that should be added with 30 data points
            self.assertIn('rsi_14', train_df.columns)
            self.assertIn('bb_upper_20', train_df.columns)
            self.assertIn('bb_middle_20', train_df.columns)
            self.assertIn('bb_lower_20', train_df.columns)
            
            # MACD may not be added due to requiring more data points
            # self.assertIn('macd_line', train_df.columns)
            # self.assertIn('macd_signal', train_df.columns)
            # self.assertIn('macd_histogram', train_df.columns)
            
            self.assertIn('obv', train_df.columns)
            
            # Check for EMA columns (only the ones that fit within data size)
            self.assertIn('ema_5', train_df.columns)
            self.assertIn('ema_10', train_df.columns)
            self.assertIn('ema_20', train_df.columns)
            
            # EMAs that won't be added due to data size
            # Since we filter to periods < len(df), data with 30 points can have ema_20 but not ema_30
            self.assertNotIn('ema_30', train_df.columns)
            
            # Other indicators
            self.assertIn('atr_14', train_df.columns)
            
            # ROC should use existing close_roc
            self.assertIn('close_roc', train_df.columns)
            self.assertNotIn('roc_12', train_df.columns)
            
            # Method should be chainable
            self.assertIs(features.add_indicators(), features)
    
    def test_scale_features(self):
        """Test that scale_features creates production_features with scaled values."""
        # Patch any print statements to avoid cluttering test output
        with patch('builtins.print'):
            features = Features(self.data)
            features.add_indicators()
            features.scale_features()
            
            # Check that production_features contains all splits
            self.assertEqual(set(features.production_features.keys()), set(features.input_data.keys()))
            
            # Check that production_features has the same columns as input_data
            self.assertEqual(
                set(features.production_features['train'].columns), 
                set(features.input_data['train'].columns)
            )
            
            # Check that numeric columns were scaled (values changed)
            train_before = features.input_data['train']
            train_after = features.production_features['train']
            
            # Check a few columns to ensure they've been scaled
            for col in ['close', 'high', 'low', 'rsi_14']:
                self.assertFalse(
                    (train_before[col] == train_after[col]).all(),
                    f"Column {col} was not scaled"
                )
            
            # Method should be chainable
            self.assertIs(features.scale_features(), features)
    
    def test_process_features(self):
        """Test that process_features combines add_indicators and scale_features."""
        features = Features(self.data)
        
        # Mock the methods to check they're called
        with patch.object(features, 'add_indicators', return_value=features) as mock_add:
            with patch.object(features, 'scale_features', return_value=features) as mock_scale:
                result = features.process_features()
                
                # Check that both methods were called
                mock_add.assert_called_once()
                mock_scale.assert_called_once()
                
                # Check that result is chainable
                self.assertIs(result, features)
    
    def test_legacy_methods(self):
        """Test that legacy methods work with the new implementation."""
        features = Features(self.data)
        
        # Test a few legacy methods
        rsi_result = features.rsi()
        bb_result = features.bollinger_bands()
        macd_result = features.macd()
        
        # Check the structure of the results
        self.assertEqual(set(rsi_result.keys()), set(features.input_data.keys()))
        self.assertEqual(set(bb_result.keys()), set(features.input_data.keys()))
        self.assertEqual(set(macd_result.keys()), set(features.input_data.keys()))
        
        # Check that BB has the correct components
        for split_name, components in bb_result.items():
            if len(components.get('upper', [])) > 0:  # Skip empty splits
                self.assertEqual(set(components.keys()), {'upper', 'middle', 'lower'})
    
    def test_with_missing_columns(self):
        """Test behavior when required columns are missing."""
        # Patch any print statements to avoid cluttering test output
        with patch('builtins.print'):
            # Create data missing 'high' and 'low'
            limited_df = self.normal_df[['open_time', 'close_time', 'open', 'close', 'volume', 'close_roc']].copy()
            data_missing_cols = MockData({
                'train': limited_df.iloc[:30].copy(),
                'test': limited_df.iloc[30:40].copy(),
                'validate': limited_df.iloc[40:].copy()
            })
            
            features = Features(data_missing_cols)
            
            # Capture output to avoid cluttering test results
            with patch('builtins.print'):
                features.add_indicators()
            
            # Check that RSI is added (only needs 'close')
            self.assertIn('rsi_14', features.input_data['train'].columns)
            
            # OBV should be added (needs 'close' and 'volume')
            self.assertIn('obv', features.input_data['train'].columns)
            
            # Indicators requiring 'high' and 'low' should not be added
            self.assertNotIn('atr_14', features.input_data['train'].columns)
            self.assertNotIn('cci_20', features.input_data['train'].columns)
            self.assertNotIn('ichimoku_conversion_line', features.input_data['train'].columns)
            self.assertNotIn('fractals_bearish_5', features.input_data['train'].columns)
            
            # Test scale_features with missing columns
            features.scale_features()
            
            # Check that production_features was created for all splits
            self.assertEqual(set(features.production_features.keys()), set(features.input_data.keys()))
            
            # Check that columns were preserved
            train_before = features.input_data['train']
            train_after = features.production_features['train']
            self.assertEqual(set(train_before.columns), set(train_after.columns))

if __name__ == '__main__':
    unittest.main() 
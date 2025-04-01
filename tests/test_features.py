import unittest
from unittest.mock import patch, MagicMock, Mock
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import RobustScaler, StandardScaler

from loop.data import HistoricalKlinesData
from loop.features import Features


class TestFeatures(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures for each test method"""
        # Create sample data with all required columns
        self.sample_data = pd.DataFrame({
            'open_time': [1672531200000, 1672617600000, 1672704000000, 1672790400000, 1672876800000],
            'close_time': [1672617599999, 1672703999999, 1672790399999, 1672876799999, 1672963199999],
            'num_trades': [100, 200, 300, 400, 500],
            'open': [10.0, 11.0, 12.0, 11.5, 11.8],
            'high': [12.0, 13.0, 14.0, 13.5, 12.5],
            'low': [9.0, 10.0, 11.0, 10.5, 11.0],
            'close': [11.0, 12.0, 13.0, 12.0, 12.2],
            'volume': [1000.0, 1100.0, 1200.0, 1000.0, 1050.0],
            'qav': [10000.0, 11000.0, 12000.0, 10500.0, 10800.0],
            'taker_base_vol': [500.0, 550.0, 600.0, 520.0, 540.0],
            'taker_quote_vol': [5500.0, 6050.0, 6600.0, 6240.0, 6588.0],
            'ignore': [0.0, 0.0, 0.0, 0.0, 0.0]
        })
        
        # Convert open_time to datetime for proper processing
        self.sample_data['open_time'] = pd.to_datetime(self.sample_data['open_time'], unit='ms')
        
        # Create a HistoricalKlinesData instance with the sample data
        self.hkd = HistoricalKlinesData(data=self.sample_data)
        
        # Split the data to create input_data attribute
        self.hkd.split_data(train_ratio=2, test_ratio=2, validate_ratio=1)
        
        # Create data with missing required columns
        self.data_missing_columns = pd.DataFrame({
            'open_time': [1672531200000, 1672617600000, 1672704000000],
            'close': [11.0, 12.0, 13.0],
            'volume': [1000.0, 1100.0, 1200.0]
        })
        
        # Create a HistoricalKlinesData-like object with missing columns
        self.hkd_missing_columns = MagicMock()
        self.hkd_missing_columns.data = self.data_missing_columns
        self.hkd_missing_columns.input_data = {
            'train': self.data_missing_columns.iloc[:1].copy(),
            'test': self.data_missing_columns.iloc[1:2].copy(),
            'validate': self.data_missing_columns.iloc[2:].copy()
        }
        
        # Create a HistoricalKlinesData-like object with empty datasets
        self.hkd_empty = MagicMock()
        self.hkd_empty.data = pd.DataFrame(columns=self.sample_data.columns)
        self.hkd_empty.input_data = {
            'train': pd.DataFrame(columns=self.sample_data.columns),
            'test': pd.DataFrame(columns=self.sample_data.columns),
            'validate': pd.DataFrame(columns=self.sample_data.columns)
        }
        
        # Create a HistoricalKlinesData-like object without input_data
        self.hkd_no_input_data = MagicMock()
        self.hkd_no_input_data.data = self.sample_data
        self.hkd_no_input_data.input_data = {}
        
        # Create mock for ta.add_all_ta_features
        # Add all necessary indicator columns that would be produced by ta
        self.ta_result = self.sample_data.copy()
        # Add momentum indicators
        self.ta_result['momentum_rsi'] = [70.0, 65.0, 60.0, 55.0, 50.0]
        self.ta_result['momentum_stoch'] = [80.0, 75.0, 70.0, 65.0, 60.0]
        self.ta_result['momentum_stoch_signal'] = [85.0, 80.0, 75.0, 70.0, 65.0]
        self.ta_result['momentum_stoch_rsi'] = [90.0, 85.0, 80.0, 75.0, 70.0]
        self.ta_result['momentum_stoch_rsi_k'] = [95.0, 90.0, 85.0, 80.0, 75.0]
        self.ta_result['momentum_stoch_rsi_d'] = [92.0, 87.0, 82.0, 77.0, 72.0]
        self.ta_result['momentum_wr'] = [-20.0, -25.0, -30.0, -35.0, -40.0]
        
        # Add volatility indicators
        self.ta_result['volatility_bbp'] = [0.8, 0.7, 0.6, 0.5, 0.4]
        self.ta_result['volatility_bbhi'] = [1.0, 0.0, 1.0, 0.0, 1.0]
        self.ta_result['volatility_bbli'] = [0.0, 1.0, 0.0, 1.0, 0.0]
        self.ta_result['volatility_kcp'] = [0.7, 0.6, 0.5, 0.4, 0.3]
        self.ta_result['volatility_kchi'] = [1.0, 0.0, 1.0, 0.0, 1.0]
        self.ta_result['volatility_kcli'] = [0.0, 1.0, 0.0, 1.0, 0.0]
        self.ta_result['volatility_dcp'] = [0.6, 0.5, 0.4, 0.3, 0.2]
        
        # Add trend indicators
        self.ta_result['trend_macd'] = [0.5, 0.4, 0.3, 0.2, 0.1]
        self.ta_result['trend_macd_signal'] = [0.45, 0.35, 0.25, 0.15, 0.05]
        self.ta_result['trend_macd_diff'] = [0.05, 0.05, 0.05, 0.05, 0.05]
        self.ta_result['trend_ema_fast'] = [11.5, 12.5, 13.5, 12.5, 12.3]
        self.ta_result['trend_ema_slow'] = [11.0, 12.0, 13.0, 12.0, 12.1]
        self.ta_result['trend_adx'] = [30.0, 32.0, 34.0, 36.0, 38.0]
        self.ta_result['trend_adx_pos'] = [20.0, 22.0, 24.0, 26.0, 28.0]
        self.ta_result['trend_adx_neg'] = [10.0, 12.0, 14.0, 16.0, 18.0]
        self.ta_result['trend_stc'] = [50.0, 55.0, 60.0, 65.0, 70.0]
        self.ta_result['trend_psar_up'] = [0.0, 1.0, 0.0, 1.0, 0.0]
        self.ta_result['trend_psar_down'] = [1.0, 0.0, 1.0, 0.0, 1.0]
        self.ta_result['trend_psar_up_indicator'] = [0.0, 1.0, 0.0, 1.0, 0.0]
        self.ta_result['trend_psar_down_indicator'] = [1.0, 0.0, 1.0, 0.0, 1.0]
        self.ta_result['trend_visual_ichimoku_a'] = [11.2, 12.2, 13.2, 12.2, 12.3]
        self.ta_result['trend_visual_ichimoku_b'] = [11.3, 12.3, 13.3, 12.3, 12.4]
        
        # Add volume indicators
        self.ta_result['volume_adi'] = [10000.0, 11000.0, 12000.0, 13000.0, 14000.0]
        self.ta_result['volume_obv'] = [5000.0, 6000.0, 7000.0, 8000.0, 9000.0]
        self.ta_result['volume_cmf'] = [0.1, 0.2, 0.3, 0.4, 0.5]
        self.ta_result['volume_vpt'] = [1000.0, 1100.0, 1200.0, 1300.0, 1400.0]
        
        # Add Numeric column that should be scaled
        self.ta_result['numeric_feature'] = [1.0, 2.0, 3.0, 4.0, 5.0]
        
    def test_init_valid_data(self):
        """Test initialization with valid data"""
        features = Features(self.hkd)
        
        self.assertEqual(features.data.equals(self.hkd.data), True)
        self.assertEqual(len(features.input_data), 3)  # train, test, validate
        self.assertEqual(features.scaler_type, 'robust')
        self.assertEqual(features.live_data, False)
        
    def test_init_invalid_scaler(self):
        """Test initialization with invalid scaler type"""
        with self.assertRaises(ValueError) as context:
            Features(self.hkd, scaler_type='invalid')
        self.assertIn("scaler_type must be 'robust' or 'standard'", str(context.exception))
        
    def test_init_unsplit_data(self):
        """Test initialization with data that hasn't been split"""
        with self.assertRaises(ValueError) as context:
            Features(self.hkd_no_input_data)
        self.assertIn("Data must be split before features can be created", str(context.exception))
        
    @patch('ta.add_all_ta_features')
    def test_add_features(self, mock_ta):
        """Test add_features method"""
        # Mock ta.add_all_ta_features to return our predefined result
        mock_ta.return_value = self.ta_result
        
        features = Features(self.hkd)
        result = features.add_features()
        
        # Verify mock was called with correct parameters
        mock_ta.assert_called()
        
        # Verify result is the same instance (returns self)
        self.assertIs(result, features)
        
        # Verify that columns are dropped as expected
        for split_name in features.production_features.keys():
            if len(features.production_features[split_name]) > 0:
                self.assertNotIn('trend_stc', features.production_features[split_name].columns)
                self.assertNotIn('trend_psar_up', features.production_features[split_name].columns)
                self.assertNotIn('trend_psar_down', features.production_features[split_name].columns)
        
    @patch('ta.add_all_ta_features')
    def test_add_indicators(self, mock_ta):
        """Test _add_indicators method"""
        # Mock ta.add_all_ta_features to return our predefined result
        mock_ta.return_value = self.ta_result
        
        features = Features(self.hkd)
        result = features._add_indicators()
        
        # Verify mock was called with correct parameters
        mock_ta.assert_called_with(
            df=self.hkd.data,
            open='open',
            close='close',
            high='high',
            low='low',
            volume='volume'
        )
        
        # Verify result is the same instance (returns self)
        self.assertIs(result, features)
        
        # Verify that ROC was added
        for split_name, df in features.input_data.items():
            if len(df) > 0:
                self.assertIn('close_roc', df.columns)
                
    def test_add_indicators_missing_columns(self):
        """Test _add_indicators method with missing required columns"""
        features = Features(self.hkd_missing_columns)
        
        with self.assertRaises(ValueError) as context:
            features._add_indicators()
        self.assertIn("Required columns", str(context.exception))
        
    def test_scale_features_robust(self):
        """Test _scale_features method with robust scaler"""
        # Create mock data with only the numeric feature for simplicity
        mock_data = MagicMock()
        mock_data.data = self.ta_result.copy()
        
        # Create input_data with train/test/validate splits
        mock_data.input_data = {
            'train': self.ta_result.iloc[:2].copy(),
            'test': self.ta_result.iloc[2:4].copy(),
            'validate': self.ta_result.iloc[4:].copy()
        }
        
        features = Features(mock_data, scaler_type='robust')
        
        # Replace _get_feature_columns to return a known set of columns
        features._get_feature_columns = MagicMock(return_value=['numeric_feature'])
        
        result = features._scale_features()
        
        # Verify result is the same instance (returns self)
        self.assertIs(result, features)
        
        # Verify scaler was created
        self.assertIsInstance(features.scaler, RobustScaler)
        
        # Verify production_features was created for each split
        self.assertEqual(len(features.production_features), 3)
        
    def test_scale_features_standard(self):
        """Test _scale_features method with standard scaler"""
        # Create mock data with only the numeric feature for simplicity
        mock_data = MagicMock()
        mock_data.data = self.ta_result.copy()
        
        # Create input_data with train/test/validate splits
        mock_data.input_data = {
            'train': self.ta_result.iloc[:2].copy(),
            'test': self.ta_result.iloc[2:4].copy(),
            'validate': self.ta_result.iloc[4:].copy()
        }
        
        features = Features(mock_data, scaler_type='standard')
        
        # Replace _get_feature_columns to return a known set of columns
        features._get_feature_columns = MagicMock(return_value=['numeric_feature'])
        
        result = features._scale_features()
        
        # Verify result is the same instance (returns self)
        self.assertIs(result, features)
        
        # Verify scaler was created
        self.assertIsInstance(features.scaler, StandardScaler)
        
        # Verify production_features was created for each split
        self.assertEqual(len(features.production_features), 3)
        
    def test_get_feature_columns(self):
        """Test _get_feature_columns method"""
        features = Features(self.hkd)
        
        # Create a test DataFrame with various column types
        test_df = pd.DataFrame({
            'numeric1': [1.0, 2.0],
            'numeric2': [3, 4],
            'categorical': ['A', 'B'],
            'open': [10.0, 11.0],  # Should be excluded
            'high': [12.0, 13.0],  # Should be excluded
            'momentum_rsi': [70.0, 75.0],  # Should be excluded
        })
        
        feature_cols = features._get_feature_columns(test_df)
        
        # Verify numeric columns not in exclude_cols are included
        self.assertIn('numeric1', feature_cols)
        self.assertIn('numeric2', feature_cols)
        
        # Verify columns in exclude_cols are not included
        self.assertNotIn('open', feature_cols)
        self.assertNotIn('high', feature_cols)
        self.assertNotIn('momentum_rsi', feature_cols)
        
        # Verify non-numeric columns are not included
        self.assertNotIn('categorical', feature_cols)
        
    def test_check_required_columns_valid(self):
        """Test _check_required_columns method with valid columns"""
        features = Features(self.hkd)
        
        # This should not raise an error
        features._check_required_columns(['open', 'high', 'low', 'close', 'volume'])
        
    def test_check_required_columns_missing(self):
        """Test _check_required_columns method with missing columns"""
        features = Features(self.hkd_missing_columns)
        
        with self.assertRaises(ValueError) as context:
            features._check_required_columns(['open', 'high', 'low', 'close', 'volume'])
        self.assertIn("Required columns", str(context.exception))
        
    def test_check_required_columns_empty(self):
        """Test _check_required_columns method with empty datasets"""
        features = Features(self.hkd_empty)
        
        with self.assertRaises(ValueError) as context:
            features._check_required_columns(['open', 'high', 'low', 'close', 'volume'])
        self.assertIn("All datasets are empty", str(context.exception))
        
    def test_has_required_columns_true(self):
        """Test _has_required_columns method when columns exist"""
        features = Features(self.hkd)
        
        # This should return True
        result = features._has_required_columns(['open', 'high', 'low', 'close', 'volume'])
        self.assertTrue(result)
        
    def test_has_required_columns_false(self):
        """Test _has_required_columns method when columns are missing"""
        features = Features(self.hkd_missing_columns)
        
        # This should return False
        result = features._has_required_columns(['open', 'high', 'low', 'close', 'volume'])
        self.assertFalse(result)
        
    def test_live_data_flag(self):
        """Test the live_data flag"""
        features = Features(self.hkd, live_data=True)
        self.assertTrue(features.live_data)
        
        features = Features(self.hkd, live_data=False)
        self.assertFalse(features.live_data)


if __name__ == '__main__':
    unittest.main() 
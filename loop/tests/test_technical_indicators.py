import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from loop.features import Features

class MockData:
    """Mock Data class for testing Features."""
    def __init__(self, input_data=None):
        self.input_data = input_data or {}

class TestTechnicalIndicators(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        # Create a sample DataFrame with OHLCV data for testing indicators
        self.sample_df = pd.DataFrame({
            'open_time': list(range(100, 160)),  # Increase data points for indicators requiring longer periods
            'close_time': list(range(101, 161)),
            'open': [100 + i * 0.5 for i in range(60)],
            'high': [105 + i * 0.5 for i in range(60)],
            'low': [95 + i * 0.5 for i in range(60)],
            'close': [102 + i * 0.5 for i in range(60)],
            'volume': [1000 + i * 10 for i in range(60)],
            'close_roc': [0.01 * (i % 5 - 2) for i in range(60)]  # Alternating values
        })
        
        # Create mock Data object with train, test, validate splits
        self.data = MockData({
            'train': self.sample_df.iloc[:40].copy(),
            'test': self.sample_df.iloc[40:50].copy(),
            'validate': self.sample_df.iloc[50:].copy()
        })
        
        # Create a Features object with our test data
        # Use patches to bypass the real scaling
        with patch('features.RobustScaler') as mock_scaler_class:
            # Set up the mock scaler
            mock_scaler = MagicMock()
            mock_scaler.fit.return_value = None
            mock_scaler.transform.return_value = lambda x: x
            mock_scaler_class.return_value = mock_scaler
            
            # Create feature instance and patch methods to avoid actual calculations
            with patch.object(Features, '_fit_transform_data'):
                # Create instance but don't call _fit_transform_data
                self.features = Features(self.data)
                # Set scaled_data directly
                self.features.scaled_data = self.data.input_data
                # Set scaler
                self.features.scaler = mock_scaler
    
    def test_check_required_columns(self):
        """Test that _check_required_columns correctly validates columns."""
        # Should not raise for existing columns
        self.features._check_required_columns(['open', 'close'])
        
        # Should raise for missing columns
        with self.assertRaises(ValueError):
            self.features._check_required_columns(['missing_column'])
    
    def test_calculate_for_all_splits(self):
        """Test that _calculate_for_all_splits applies function to all splits."""
        # Define a simple function to apply
        def double_close(df):
            return df['close'].values * 2
        
        # Apply to all splits
        result = self.features._calculate_for_all_splits(double_close)
        
        # Check that all splits are present
        self.assertEqual(set(result.keys()), {'train', 'test', 'validate'})
        
        # Check that function was applied correctly to each split
        for split_name, df in self.data.input_data.items():
            expected = df['close'].values * 2
            np.testing.assert_array_almost_equal(result[split_name], expected)
    
    def test_helper_calculations(self):
        """Test the helper calculation methods."""
        df = self.data.input_data['train']
        
        # Test SMA
        sma = self.features._calculate_sma(df, 'close', 5)
        self.assertEqual(len(sma), len(df))
        self.assertTrue(np.isnan(sma[0]))  # First values should be NaN
        self.assertFalse(np.isnan(sma[5]))  # Later values should be calculated
        
        # Test EMA
        ema = self.features._calculate_ema(df, 'close', 5)
        self.assertEqual(len(ema), len(df))
        self.assertFalse(np.isnan(ema[0]))  # EMA starts with first value
        
        # Test STD
        std = self.features._calculate_std(df, 'close', 5)
        self.assertEqual(len(std), len(df))
        self.assertTrue(np.isnan(std[0]))  # First values should be NaN
        self.assertFalse(np.isnan(std[5]))  # Later values should be calculated
    
    @patch('loop.features.ta.momentum.RSIIndicator')
    def test_rsi(self, mock_rsi_class):
        """Test RSI calculation."""
        # Set up mock for RSI
        mock_rsi = MagicMock()
        mock_rsi.rsi.return_value = pd.Series(np.linspace(0, 100, len(self.data.input_data['train'])))
        mock_rsi_class.return_value = mock_rsi
        
        # Test with default parameters
        rsi_result = self.features.rsi()
        
        # Check structure
        self.assertEqual(set(rsi_result.keys()), {'train', 'test', 'validate'})
        
        # Test with custom period
        self.features.rsi(period=7)
        # Verify the period was used
        for call in mock_rsi_class.call_args_list:
            args, kwargs = call
            self.assertIn('window', kwargs)
    
    @patch('loop.features.ta.volatility.BollingerBands')
    def test_bollinger_bands(self, mock_bb_class):
        """Test Bollinger Bands calculation."""
        # Set up mock for Bollinger Bands
        mock_bb = MagicMock()
        n = len(self.data.input_data['train'])
        mock_bb.bollinger_hband.return_value = pd.Series(np.linspace(110, 150, n))
        mock_bb.bollinger_mavg.return_value = pd.Series(np.linspace(100, 140, n))
        mock_bb.bollinger_lband.return_value = pd.Series(np.linspace(90, 130, n))
        mock_bb_class.return_value = mock_bb
        
        # Test with default parameters
        bb_result = self.features.bollinger_bands()
        
        # Check structure
        self.assertEqual(set(bb_result.keys()), {'train', 'test', 'validate'})
        
        # Check components
        for components in bb_result.values():
            self.assertEqual(set(components.keys()), {'upper', 'middle', 'lower'})
        
        # Test with custom parameters
        self.features.bollinger_bands(period=10, std_dev=1.5)
        # Verify the parameters were used
        for call in mock_bb_class.call_args_list:
            args, kwargs = call
            self.assertIn('window', kwargs)
            self.assertIn('window_dev', kwargs)
    
    @patch('loop.features.ta.trend.MACD')
    def test_macd(self, mock_macd_class):
        """Test MACD calculation."""
        # Set up mock for MACD
        mock_macd = MagicMock()
        n = len(self.data.input_data['train'])
        mock_macd.macd.return_value = pd.Series(np.linspace(0, 10, n))
        mock_macd.macd_signal.return_value = pd.Series(np.linspace(0, 8, n))
        mock_macd.macd_diff.return_value = pd.Series(np.linspace(0, 2, n))
        mock_macd_class.return_value = mock_macd
        
        # Test with default parameters
        macd_result = self.features.macd()
        
        # Check structure
        self.assertEqual(set(macd_result.keys()), {'train', 'test', 'validate'})
        
        # Check components
        for components in macd_result.values():
            self.assertEqual(set(components.keys()), {'macd', 'signal', 'histogram'})
        
        # Test with custom parameters
        self.features.macd(fast_period=8, slow_period=21, signal_period=5)
        # Verify the parameters were used
        for call in mock_macd_class.call_args_list:
            args, kwargs = call
            self.assertIn('window_fast', kwargs)
            self.assertIn('window_slow', kwargs)
            self.assertIn('window_sign', kwargs)
    
    def test_ema(self):
        """Test EMA calculation."""
        # Patch the internal calculation method
        with patch.object(self.features, '_calculate_ema') as mock_ema:
            mock_ema.return_value = np.zeros(len(self.data.input_data['train']))
            
            # Test with default parameters
            ema_result = self.features.ema()
            
            # Check structure
            self.assertEqual(set(ema_result.keys()), {'train', 'test', 'validate'})
            
            # Reset the mock to clear call count
            mock_ema.reset_mock()
            
            # Test with custom periods
            custom_periods = [3, 15, 30]
            self.features.ema(periods=custom_periods)
            
            # Verify the calculation was called the right number of times
            # Each period is calculated for each split
            self.assertEqual(mock_ema.call_count, len(custom_periods) * len(self.data.input_data))
    
    @patch('loop.features.ta.volatility.AverageTrueRange')
    def test_atr(self, mock_atr_class):
        """Test ATR calculation."""
        # Set up mock for ATR
        mock_atr = MagicMock()
        mock_atr.average_true_range.return_value = pd.Series(np.ones(len(self.data.input_data['train'])))
        mock_atr_class.return_value = mock_atr
        
        # Test with default parameters
        atr_result = self.features.atr()
        
        # Check structure
        self.assertEqual(set(atr_result.keys()), {'train', 'test', 'validate'})
        
        # Test with custom period
        self.features.atr(period=7)
        # Verify the period was used
        for call in mock_atr_class.call_args_list:
            args, kwargs = call
            self.assertIn('window', kwargs)
    
    @patch('loop.features.ta.volume.OnBalanceVolumeIndicator')
    def test_obv(self, mock_obv_class):
        """Test OBV calculation."""
        # Set up mock for OBV
        mock_obv = MagicMock()
        mock_obv.on_balance_volume.return_value = pd.Series(np.ones(len(self.data.input_data['train'])))
        mock_obv_class.return_value = mock_obv
        
        # Test function call
        obv_result = self.features.obv()
        
        # Check structure
        self.assertEqual(set(obv_result.keys()), {'train', 'test', 'validate'})
    
    def test_roc(self):
        """Test ROC calculation with existing close_roc column."""
        # Test with default parameters
        roc_result = self.features.roc()
        
        # Check structure
        self.assertEqual(set(roc_result.keys()), {'train', 'test', 'validate'})
        
        # Check values - should use existing close_roc
        for split_name, arr in roc_result.items():
            np.testing.assert_array_almost_equal(
                arr, 
                self.data.input_data[split_name]['close_roc'].values
            )
    
    @patch('loop.features.ta.momentum.ROCIndicator')
    def test_roc_calculated(self, mock_roc_class):
        """Test ROC calculation when close_roc doesn't exist."""
        # Set up mock for ROC
        mock_roc = MagicMock()
        mock_roc.roc.return_value = pd.Series(np.ones(len(self.data.input_data['train'])))
        mock_roc_class.return_value = mock_roc
        
        # Create a new data object without close_roc
        data_no_roc = MockData({
            'train': self.sample_df.drop(columns=['close_roc']).iloc[:5].copy()
        })
        
        # Create features instance with mocks to bypass actual scaling
        with patch('features.RobustScaler'), patch.object(Features, '_fit_transform_data'):
            features_no_roc = Features(data_no_roc)
            features_no_roc.scaled_data = data_no_roc.input_data
            
            # Patch the _check_required_columns method to avoid errors
            with patch.object(features_no_roc, '_check_required_columns'):
                # Calculate ROC
                result = features_no_roc.roc()
                
                # Verify ROC indicator was created
                mock_roc_class.assert_called_once()
    
    @patch('loop.features.ta.trend.CCIIndicator')
    def test_cci(self, mock_cci_class):
        """Test CCI calculation."""
        # Set up mock for CCI
        mock_cci = MagicMock()
        mock_cci.cci.return_value = pd.Series(np.ones(len(self.data.input_data['train'])))
        mock_cci_class.return_value = mock_cci
        
        # Test with default parameters
        cci_result = self.features.cci()
        
        # Check structure
        self.assertEqual(set(cci_result.keys()), {'train', 'test', 'validate'})
        
        # Test with custom period
        self.features.cci(period=10)
        # Verify the period was used
        for call in mock_cci_class.call_args_list:
            args, kwargs = call
            self.assertIn('window', kwargs)
    
    @patch('loop.features.ta.trend.IchimokuIndicator')
    def test_ichimoku(self, mock_ichimoku_class):
        """Test Ichimoku Cloud calculation."""
        # Set up mock for Ichimoku
        mock_ichimoku = MagicMock()
        n = len(self.data.input_data['train'])
        mock_ichimoku.ichimoku_conversion_line.return_value = pd.Series(np.ones(n))
        mock_ichimoku.ichimoku_base_line.return_value = pd.Series(np.ones(n))
        mock_ichimoku.ichimoku_a.return_value = pd.Series(np.ones(n))
        mock_ichimoku.ichimoku_b.return_value = pd.Series(np.ones(n))
        mock_ichimoku_class.return_value = mock_ichimoku
        
        # Test function call
        ichi_result = self.features.ichimoku()
        
        # Check structure
        self.assertEqual(set(ichi_result.keys()), {'train', 'test', 'validate'})
        
        # Check components
        for components in ichi_result.values():
            self.assertEqual(set(components.keys()), 
                          {'conversion_line', 'base_line', 'a', 'b'})
    
    def test_williams_fractals(self):
        """Test Williams Fractals calculation with mocked calculation function."""
        # Patch the internal calculation method to avoid complexity
        with patch.object(self.features, '_calculate_for_all_splits') as mock_calc:
            # Define a mock result
            mock_result = {
                'train': {'bearish': np.zeros(40), 'bullish': np.zeros(40)},
                'test': {'bearish': np.zeros(10), 'bullish': np.zeros(10)},
                'validate': {'bearish': np.zeros(10), 'bullish': np.zeros(10)}
            }
            mock_calc.return_value = mock_result
            
            # Test with default parameters
            wf_result = self.features.williams_fractals()
            
            # Check structure
            self.assertEqual(set(wf_result.keys()), {'train', 'test', 'validate'})
            
            # Reset mock to clear call count
            mock_calc.reset_mock()
            
            # Test with custom period
            self.features.williams_fractals(period=3)
            self.assertEqual(mock_calc.call_count, 1)
    
    def test_missing_required_columns(self):
        """Test indicators when required columns are missing."""
        # Create data missing 'high' and 'low'
        data_missing_cols = MockData({
            'train': self.sample_df[['open_time', 'close_time', 'open', 'close', 'volume']].iloc[:5].copy()
        })
        
        # Create features instance with mocks to bypass actual scaling
        with patch('features.RobustScaler'), patch.object(Features, '_fit_transform_data'):
            features_missing_cols = Features(data_missing_cols)
            features_missing_cols.scaled_data = data_missing_cols.input_data
            
            # Test indicators that need 'high' and 'low'
            with self.assertRaises(ValueError):
                with patch.object(features_missing_cols, '_check_required_columns', 
                                wraps=features_missing_cols._check_required_columns):
                    features_missing_cols.atr()
            
            with self.assertRaises(ValueError):
                with patch.object(features_missing_cols, '_check_required_columns',
                                wraps=features_missing_cols._check_required_columns):
                    features_missing_cols.cci()
            
            with self.assertRaises(ValueError):
                with patch.object(features_missing_cols, '_check_required_columns',
                                wraps=features_missing_cols._check_required_columns):
                    features_missing_cols.ichimoku()
            
            with self.assertRaises(ValueError):
                with patch.object(features_missing_cols, '_check_required_columns',
                                wraps=features_missing_cols._check_required_columns):
                    features_missing_cols.williams_fractals()
            
            # These should not raise errors for missing 'high' and 'low'
            with patch('loop.features.ta.momentum.RSIIndicator'):
                features_missing_cols.rsi()  # Only needs 'close'
            
            with patch('loop.features.ta.volume.OnBalanceVolumeIndicator'):
                features_missing_cols.obv()  # Needs 'close' and 'volume'

if __name__ == '__main__':
    unittest.main() 
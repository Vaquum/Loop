import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import pytest
from datetime import datetime
from loop.data import HistoricalKlinesData


class TestHistoricalKlinesData(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures for each test method"""
        # Create sample data with all required columns
        self.sample_data = pd.DataFrame({
            'open_time': [datetime(2023, 1, 1), datetime(2023, 1, 2), datetime(2023, 1, 3)],
            'close_time': [1672531200000, 1672617600000, 1672704000000],
            'num_trades': [100, 200, 300],
            'open': [10.0, 11.0, 12.0],
            'high': [12.0, 13.0, 14.0],
            'low': [9.0, 10.0, 11.0],
            'close': [11.0, 12.0, 13.0],
            'volume': [1000.0, 1100.0, 1200.0],
            'qav': [10000.0, 11000.0, 12000.0],
            'taker_base_vol': [500.0, 550.0, 600.0],
            'taker_quote_vol': [5500.0, 6050.0, 6600.0],
            'ignore': [0.0, 0.0, 0.0]
        })
        
        # Convert open_time back to int to match expected input format
        self.sample_data_with_int_time = self.sample_data.copy()
        self.sample_data_with_int_time['open_time'] = [1672531200000, 1672617600000, 1672704000000]
        
        # Sample data with NA values
        self.sample_data_with_na = self.sample_data.copy()
        self.sample_data_with_na.loc[1, 'close'] = np.nan
        
        # Sample data with invalid columns
        self.sample_data_invalid_cols = pd.DataFrame({
            'open_time': [1672531200000, 1672617600000, 1672704000000],
            'close': [11.0, 12.0, 13.0],
            'invalid_column': [1, 2, 3]
        })
        
        # Unsorted data for testing chronological validation
        self.unsorted_data = self.sample_data.copy()
        self.unsorted_data['open_time'] = [datetime(2023, 1, 3), datetime(2023, 1, 1), datetime(2023, 1, 2)]

    def test_init_with_dataframe(self):
        """Test initialization with a pandas DataFrame"""
        hkd = HistoricalKlinesData(data=self.sample_data_with_int_time)
        
        # Verify data is loaded and processed correctly
        self.assertEqual(len(hkd.data), 3)
        self.assertTrue(isinstance(hkd.data['open_time'][0], pd.Timestamp))
        self.assertEqual(hkd.data['high'][1], 13.0)
        
    def test_init_with_dataframe_drop_na_false(self):
        """Test initialization with drop_na=False"""
        hkd = HistoricalKlinesData(data=self.sample_data_with_na, drop_na=False)
        
        # Verify NA values are retained
        self.assertEqual(len(hkd.data), 3)
        self.assertTrue(np.isnan(hkd.data['close'][1]))
        
    def test_init_with_dataframe_drop_na_true(self):
        """Test initialization with drop_na=True"""
        hkd = HistoricalKlinesData(data=self.sample_data_with_na, drop_na=True)
        
        # Verify NA values are dropped
        self.assertEqual(len(hkd.data), 2)
        
    @patch('pandas.read_csv')
    def test_init_with_file_path(self, mock_read_csv):
        """Test initialization with a file path"""
        mock_read_csv.return_value = self.sample_data_with_int_time
        
        hkd = HistoricalKlinesData(data_file_path='dummy_path.csv')
        
        # Verify read_csv was called and data is loaded
        mock_read_csv.assert_called_once_with('dummy_path.csv')
        self.assertEqual(len(hkd.data), 3)
        
    @patch('loop.data.get_klines_historical')
    def test_init_with_api_params(self, mock_get_klines):
        """Test initialization with API parameters"""
        mock_get_klines.return_value = self.sample_data_with_int_time
        
        hkd = HistoricalKlinesData(
            data_start_date='2023-01-01',
            data_end_date='2023-01-03',
            data_interval='1d'
        )
        
        # Verify get_klines_historical was called with correct params
        mock_get_klines.assert_called_once_with('1d', '2023-01-01', '2023-01-03')
        self.assertEqual(len(hkd.data), 3)
        
    def test_init_invalid_input(self):
        """Test error is raised for invalid input"""
        with self.assertRaises(ValueError):
            HistoricalKlinesData()
            
    def test_column_validation(self):
        """Test column validation logic"""
        with self.assertRaises(AssertionError):
            HistoricalKlinesData(data=self.sample_data_invalid_cols)
            
    def test_split_data_normal(self):
        """Test normal split behavior with different ratios"""
        hkd = HistoricalKlinesData(data=self.sample_data_with_int_time)
        
        # Test default 1:1:1 split
        result = hkd.split_data()
        self.assertEqual(len(result.input_data['train']), 1)
        self.assertEqual(len(result.input_data['test']), 1)
        self.assertEqual(len(result.input_data['validate']), 1)
        
        # Test custom ratio split
        result = hkd.split_data(train_ratio=2, test_ratio=1, validate_ratio=0)
        self.assertEqual(len(result.input_data['train']), 2)
        self.assertEqual(len(result.input_data['test']), 1)
        self.assertEqual(len(result.input_data['validate']), 0)
        
    def test_split_data_edge_cases(self):
        """Test split behavior with edge case ratios"""
        hkd = HistoricalKlinesData(data=self.sample_data_with_int_time)
        
        # Test with one zero ratio
        result = hkd.split_data(train_ratio=0, test_ratio=1, validate_ratio=2)
        self.assertEqual(len(result.input_data['train']), 0)
        self.assertEqual(len(result.input_data['test']), 1)
        self.assertEqual(len(result.input_data['validate']), 2)
        
        # Test with two zero ratios
        result = hkd.split_data(train_ratio=3, test_ratio=0, validate_ratio=0)
        self.assertEqual(len(result.input_data['train']), 3)
        self.assertEqual(len(result.input_data['test']), 0)
        self.assertEqual(len(result.input_data['validate']), 0)
        
    def test_split_data_empty_dataset(self):
        """Test error for empty dataset"""
        empty_df = pd.DataFrame(columns=self.sample_data.columns)
        hkd = HistoricalKlinesData(data=empty_df)
        
        with self.assertRaises(ValueError) as context:
            hkd.split_data()
        self.assertIn('Cannot split an empty dataset', str(context.exception))
        
    def test_split_data_invalid_ratios(self):
        """Test error for invalid ratio types"""
        hkd = HistoricalKlinesData(data=self.sample_data_with_int_time)
        
        with self.assertRaises(ValueError) as context:
            hkd.split_data(train_ratio=-1, test_ratio=1, validate_ratio=1)
        self.assertIn('All ratios must be non-negative integers', str(context.exception))
        
        with self.assertRaises(ValueError) as context:
            hkd.split_data(train_ratio=1.5, test_ratio=1, validate_ratio=1)
        self.assertIn('All ratios must be non-negative integers', str(context.exception))
        
    def test_split_data_all_zero_ratios(self):
        """Test error when all ratios are zero"""
        hkd = HistoricalKlinesData(data=self.sample_data_with_int_time)
        
        with self.assertRaises(ValueError) as context:
            hkd.split_data(train_ratio=0, test_ratio=0, validate_ratio=0)
        self.assertIn('At least one ratio must be positive', str(context.exception))
        
    def test_split_data_too_few_rows(self):
        """Test error for insufficient rows"""
        # Only one row but trying to split into 3 parts
        one_row_df = self.sample_data_with_int_time.iloc[:1]
        hkd = HistoricalKlinesData(data=one_row_df)
        
        with self.assertRaises(ValueError) as context:
            hkd.split_data(train_ratio=1, test_ratio=1, validate_ratio=1)
        self.assertIn('at least 3 rows are required', str(context.exception))
        
    def test_split_data_unsorted(self):
        """Test error for unsorted data"""
        hkd = HistoricalKlinesData(data=self.unsorted_data)
        
        with self.assertRaises(ValueError) as context:
            hkd.split_data()
        self.assertIn('Data is not sorted chronologically', str(context.exception))
        
    def test_split_data_return_self(self):
        """Test that split_data returns self"""
        hkd = HistoricalKlinesData(data=self.sample_data_with_int_time)
        result = hkd.split_data()
        self.assertIs(result, hkd)


if __name__ == '__main__':
    unittest.main() 
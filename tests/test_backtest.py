import unittest
from unittest.mock import patch, MagicMock, Mock
import pandas as pd
import numpy as np
from datetime import datetime

from loop.backtest import Backtest


class TestBacktest(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures for each test method"""
        # Create sample data for testing
        self.sample_data = pd.DataFrame({
            'open_time': [
                datetime(2023, 1, 1),
                datetime(2023, 1, 2),
                datetime(2023, 1, 3)
            ],
            'open': [10000.0, 11000.0, 10500.0],
            'close': [11000.0, 10500.0, 10800.0],
            'high': [11200.0, 11100.0, 10900.0],
            'low': [9800.0, 10400.0, 10300.0],
            'volume': [100.0, 120.0, 90.0]
        })
        
        # Create an empty dataframe for empty data test
        self.empty_data = pd.DataFrame(columns=self.sample_data.columns)
        
        # Mock prep function
        self.mock_prep = MagicMock()
        self.mock_prep.return_value = np.array([
            [0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.6]
        ])
        
        # Mock predictor functions for different test scenarios
        self.mock_predictor_buy = MagicMock()
        self.mock_predictor_buy.return_value = np.array([0.9, 0.9, 0.9])  # All buy signals
        
        self.mock_predictor_hold = MagicMock()
        self.mock_predictor_hold.return_value = np.array([0.5, 0.5, 0.5])  # All hold signals
        
        self.mock_predictor_short = MagicMock()
        self.mock_predictor_short.return_value = np.array([0.1, 0.1, 0.1])  # All short signals
        
        self.mock_predictor_mixed = MagicMock()
        self.mock_predictor_mixed.return_value = np.array([0.9, 0.5, 0.1])  # Buy, hold, short
        
        # Mock account for testing
        self.mock_account = {
            'id': [0, 1],
            'action': ['hold', 'hold'],
            'position_type': ['none', 'none'],
            'timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S'), datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            'credit_usdt': [1000, 0],
            'debit_usdt': [0, 0],
            'amount_bought_btc': [0, 0],
            'amount_sold_btc': [0, 0],
            'borrowed_btc': [0, 0],
            'covered_btc': [0, 0],
            'buy_price_usdt': [0, 0],
            'sell_price_usdt': [0, 0],
            'total_usdt': [1000, 1000],
            'total_btc': [0, 0],
            'total_borrowed_btc': [0, 0]
        }
        
        # Initialize the Backtest object
        self.backtest = Backtest()
    
    @patch('pandas.cut')
    @patch('loop.backtest.Account')
    def test_run_buy_signals(self, mock_account_class, mock_pd_cut):
        """Test the run method with all 'buy' signals"""
        # Set up mock Account
        mock_account_instance = MagicMock()
        mock_account_instance.account = self.mock_account
        mock_account_class.return_value = mock_account_instance
        
        # Set up mock pd.cut to return 'buy' signals
        mock_pd_cut.return_value = pd.Series(['buy', 'buy', 'buy'])
        
        # Run the backtest
        result = self.backtest.run(
            start_usdt=1000,
            predictor=self.mock_predictor_buy,
            prep=self.mock_prep,
            data=self.sample_data
        )
        
        # Verify prep and predictor were called
        self.mock_prep.assert_called_once_with(self.sample_data, mode='predict')
        self.mock_predictor_buy.assert_called_once()
        
        # Verify Account was initialized correctly
        mock_account_class.assert_called_once_with(start_usdt=1000)
        
        # Verify account.update_account was called for each day (with buy and sell actions)
        self.assertEqual(mock_account_instance.update_account.call_count, 6)  # 3 days * 2 updates per day
        
        # Verify account.update_id was called
        self.assertEqual(mock_account_instance.update_id.call_count, 6)  # 3 days * 2 updates per day
        
        # Verify the result is the account's state
        self.assertEqual(result, self.mock_account)
        
    @patch('pandas.cut')
    @patch('loop.backtest.Account')
    def test_run_hold_signals(self, mock_account_class, mock_pd_cut):
        """Test the run method with all 'hold' signals"""
        # Set up mock Account
        mock_account_instance = MagicMock()
        mock_account_instance.account = self.mock_account
        mock_account_class.return_value = mock_account_instance
        
        # Set up mock pd.cut to return 'hold' signals
        mock_pd_cut.return_value = pd.Series(['hold', 'hold', 'hold'])
        
        # Run the backtest
        result = self.backtest.run(
            start_usdt=1000,
            predictor=self.mock_predictor_hold,
            prep=self.mock_prep,
            data=self.sample_data
        )
        
        # Verify prep and predictor were called
        self.mock_prep.assert_called_once_with(self.sample_data, mode='predict')
        self.mock_predictor_hold.assert_called_once()
        
        # Verify Account was initialized correctly
        mock_account_class.assert_called_once_with(start_usdt=1000)
        
        # Verify account.update_account was called for each day
        self.assertEqual(mock_account_instance.update_account.call_count, 3)  # 3 days * 1 update per day (hold)
        
        # Verify account.update_id was called
        self.assertEqual(mock_account_instance.update_id.call_count, 3)  # 3 days * 1 update per day
        
        # Verify the result is the account's state
        self.assertEqual(result, self.mock_account)
        
    @patch('pandas.cut')
    @patch('loop.backtest.Account')
    def test_run_short_signals(self, mock_account_class, mock_pd_cut):
        """Test the run method with all 'short' signals"""
        # Set up mock Account
        mock_account_instance = MagicMock()
        mock_account_instance.account = self.mock_account
        mock_account_class.return_value = mock_account_instance
        
        # Set up mock pd.cut to return 'short' signals
        mock_pd_cut.return_value = pd.Series(['short', 'short', 'short'])
        
        # Run the backtest
        result = self.backtest.run(
            start_usdt=1000,
            predictor=self.mock_predictor_short,
            prep=self.mock_prep,
            data=self.sample_data
        )
        
        # Verify prep and predictor were called
        self.mock_prep.assert_called_once_with(self.sample_data, mode='predict')
        self.mock_predictor_short.assert_called_once()
        
        # Verify Account was initialized correctly
        mock_account_class.assert_called_once_with(start_usdt=1000)
        
        # Verify account.update_account was called for each day
        self.assertEqual(mock_account_instance.update_account.call_count, 6)  # 3 days * 2 updates per day
        
        # Verify account.update_id was called
        self.assertEqual(mock_account_instance.update_id.call_count, 6)  # 3 days * 2 updates per day
        
        # Verify the result is the account's state
        self.assertEqual(result, self.mock_account)
        
    @patch('pandas.cut')
    @patch('loop.backtest.Account')
    def test_run_mixed_signals(self, mock_account_class, mock_pd_cut):
        """Test the run method with mixed signals"""
        # Set up mock Account
        mock_account_instance = MagicMock()
        mock_account_instance.account = self.mock_account
        mock_account_class.return_value = mock_account_instance
        
        # Set up mock pd.cut to return mixed signals
        mock_pd_cut.return_value = pd.Series(['buy', 'hold', 'short'])
        
        # Run the backtest
        result = self.backtest.run(
            start_usdt=1000,
            predictor=self.mock_predictor_mixed,
            prep=self.mock_prep,
            data=self.sample_data
        )
        
        # Verify prep and predictor were called
        self.mock_prep.assert_called_once_with(self.sample_data, mode='predict')
        self.mock_predictor_mixed.assert_called_once()
        
        # Verify Account was initialized correctly
        mock_account_class.assert_called_once_with(start_usdt=1000)
        
        # Verify account.update_account was called for each signal appropriately
        expected_calls = 5  # 'buy' (2), 'hold' (1), 'short' (2) - double-check your logic in Backtest
        self.assertEqual(mock_account_instance.update_account.call_count, expected_calls)
        
        # Verify account.update_id was called the same number of times
        self.assertEqual(mock_account_instance.update_id.call_count, expected_calls)
        
        # Verify the result is the account's state
        self.assertEqual(result, self.mock_account)
        
    @patch('pandas.cut')
    @patch('loop.backtest.Account')
    def test_run_empty_dataframe(self, mock_account_class, mock_pd_cut):
        """Test the run method with an empty dataframe"""
        # Set up mock Account
        mock_account_instance = MagicMock()
        mock_account_instance.account = self.mock_account
        mock_account_class.return_value = mock_account_instance
        
        # Set up mock pd.cut to return empty series
        mock_pd_cut.return_value = pd.Series([], dtype='object')
        
        # Run the backtest with empty data
        result = self.backtest.run(
            start_usdt=1000,
            predictor=self.mock_predictor_buy,
            prep=self.mock_prep,
            data=self.empty_data
        )
        
        # Verify account.update_account was not called (empty dataframe)
        mock_account_instance.update_account.assert_not_called()
        
        # Verify account.update_id was not called (empty dataframe)
        mock_account_instance.update_id.assert_not_called()
        
        # Verify the result is the account's state
        self.assertEqual(result, self.mock_account)
        
    @patch('pandas.cut')
    @patch('loop.backtest.Account')
    def test_init(self, mock_account_class, mock_pd_cut):
        """Test the initialization of the Backtest class"""
        # Verify that the Backtest class can be instantiated
        backtest = Backtest()
        self.assertIsInstance(backtest, Backtest)
        
    @patch('pandas.cut')
    @patch('loop.backtest.Account')
    def test_verify_account_updates(self, mock_account_class, mock_pd_cut):
        """Test specific account updates with different signals"""
        # Set up mock Account
        mock_account_instance = MagicMock()
        mock_account_instance.account = {'total_usdt': [1000]}
        mock_account_class.return_value = mock_account_instance
        
        # Set up mock pd.cut to return specific signals
        mock_pd_cut.return_value = pd.Series(['buy', 'hold', 'short'])
        
        # Run the backtest
        self.backtest.run(
            start_usdt=1000,
            predictor=self.mock_predictor_mixed,
            prep=self.mock_prep,
            data=self.sample_data
        )
        
        # Get all calls to update_account
        update_calls = mock_account_instance.update_account.call_args_list
        
        # Check the content of update_account calls based on positional arguments
        # First day - buy signal
        self.assertEqual(update_calls[0][0][0], self.sample_data.iloc[0]['open_time'])
        self.assertEqual(update_calls[0][0][1], 'buy') 
        self.assertEqual(update_calls[0][0][2], 1000)  # total_usdt
        self.assertEqual(update_calls[0][0][3], 10000.0)  # open price
        
        # First day - sell (after buy)
        self.assertEqual(update_calls[1][0][0], self.sample_data.iloc[0]['open_time'])
        self.assertEqual(update_calls[1][0][1], 'sell')
        self.assertEqual(update_calls[1][0][2], 1000)  # total_usdt 
        self.assertEqual(update_calls[1][0][3], 11000.0)  # close price
        
        # Second day - hold signal
        self.assertEqual(update_calls[2][0][0], self.sample_data.iloc[1]['open_time'])
        self.assertEqual(update_calls[2][0][1], 'hold')
        self.assertEqual(update_calls[2][0][2], 1000)  # total_usdt
        self.assertEqual(update_calls[2][0][3], 11000.0)  # open price
        
        # Third day - short signal
        self.assertEqual(update_calls[3][0][0], self.sample_data.iloc[2]['open_time'])
        self.assertEqual(update_calls[3][0][1], 'short')
        self.assertEqual(update_calls[3][0][2], 1000)  # total_usdt
        self.assertEqual(update_calls[3][0][3], 10500.0)  # open price
        
        # Third day - cover (after short)
        self.assertEqual(update_calls[4][0][0], self.sample_data.iloc[2]['open_time'])
        self.assertEqual(update_calls[4][0][1], 'cover')
        self.assertEqual(update_calls[4][0][2], 1000)  # total_usdt
        self.assertEqual(update_calls[4][0][3], 10800.0)  # close price


if __name__ == '__main__':
    unittest.main() 
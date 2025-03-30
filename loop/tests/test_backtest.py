"""
Test script for the BackTest class.

This script runs a simple test of the BackTest class with mock data and
checks that the basic functionality works as expected.
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
import tempfile
import shutil
import os
import sys

# Add the parent directory to sys.path to import loop modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Create mock classes to test BackTest
class MockPredictor:
    def predict_with_features(self, features, split_name, idx):
        # Return 0 (bearish), 1 (neutral), or 2 (bullish) based on idx
        # Create a repeating pattern for testing
        return idx % 3  

class MockFeatures:
    def __init__(self):
        # Create a mock production_features dictionary
        self.production_features = {
            'validate': self._create_mock_data()
        }
    
    def _create_mock_data(self):
        # Create a DataFrame with the required columns
        dates = pd.date_range(start='2020-01-01', periods=100)
        data = {
            'close': np.linspace(100, 200, 100),  # Linear price increase
            'high': np.linspace(105, 210, 100),   # 5% higher than close
            'low': np.linspace(95, 190, 100),     # 5% lower than close
            'volume': np.random.randint(1000, 2000, 100),
            'close_roc': np.random.normal(0.001, 0.01, 100),  # Random returns
            'high_low_ratio': np.random.uniform(1.05, 1.1, 100),
            'volume_change': np.random.normal(0, 0.05, 100),
            'high_close_ratio': np.random.uniform(0.01, 0.05, 100),
            'low_close_ratio': np.random.uniform(0.01, 0.05, 100)
        }
        return pd.DataFrame(data, index=dates)

class TestBackTest(unittest.TestCase):
    """Test the BackTest class functionality"""
    
    def setUp(self):
        """Set up the test environment"""
        from loop.loop.backtest import BackTest
        
        # Create mock objects
        self.predictor = MockPredictor()
        self.features = MockFeatures()
        
        # Create backtest instance
        self.backtester = BackTest(
            predictor=self.predictor,
            features=self.features,
            initial_capital=10000.0,
            risk_per_trade=0.02,
            stop_loss=0.02,
            take_profit=0.04,
            commission=0.001,
            slippage=0.001
        )
        
        # Create a temporary directory for test output
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up after tests"""
        # Remove temporary directory
        shutil.rmtree(self.test_dir)
    
    def test_initialization(self):
        """Test that the BackTest class initializes correctly"""
        # Check that data was loaded
        self.assertIsNotNone(self.backtester.data)
        self.assertEqual(len(self.backtester.data), 100)
    
    def test_run_backtest(self):
        """Test running a backtest"""
        # Run backtest
        results = self.backtester.run_backtest(sequence_length=10)
        
        # Check that results contain expected keys
        self.assertIn('results', results)
        self.assertIn('metrics', results)
        self.assertIn('roi', results)
        
        # Check that results have expected shape
        self.assertEqual(len(results['results']), 90)  # 100 days - 10 sequence length
        
        # Check that capital changes from initial
        final_capital = results['results']['capital'].iloc[-1]
        self.assertNotEqual(final_capital, self.backtester.initial_capital)
    
    def test_position_calculation(self):
        """Test position calculation logic"""
        # Manually calculate position size
        capital = 10000.0
        price = 150.0
        position_size = self.backtester._calculate_position_size(capital, price)
        
        # Risk per trade is 2% and stop loss is 2%
        expected_size = min(capital * 0.02 / 0.02, capital * 0.5)
        
        self.assertEqual(position_size, expected_size)
    
    def test_metrics_calculation(self):
        """Test metrics calculation"""
        # Run backtest
        self.backtester.run_backtest(sequence_length=10)
        
        # Generate report
        metrics = self.backtester.generate_report(save_dir=self.test_dir)
        
        # Check that metrics contain expected keys
        expected_metrics = [
            'roi', 'total_return', 'annual_return', 'sharpe_ratio', 
            'sortino_ratio', 'max_drawdown', 'total_trades', 'win_rate'
        ]
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
        
        # Check that report files were created
        report_files = os.listdir(self.test_dir)
        self.assertGreater(len(report_files), 0)
    
    @patch('matplotlib.pyplot.show')
    def test_plot_results(self, mock_show):
        """Test plotting backtest results (without displaying)"""
        # Mock plt.show() to prevent actual display
        
        # Run backtest
        self.backtester.run_backtest(sequence_length=10)
        
        # Plot results
        plot_path = os.path.join(self.test_dir, 'test_plot.png')
        self.backtester.plot_results(save_path=plot_path)
        
        # Check that plot file was created
        self.assertTrue(os.path.exists(plot_path))
        
        # Check that show was called
        mock_show.assert_called_once()

    def test_dashboard(self, mocker):
        """
        Test the dashboard method of the BackTest class.
        
        This test mocks the Dash app creation and verifies that the dashboard
        can be initialized correctly with backtest results.
        """
        # Mock dash.Dash class
        mock_dash = mocker.patch('dash.Dash')
        mock_app = mocker.MagicMock()
        mock_dash.return_value = mock_app
        
        # Create mock predictor and features
        mock_predictor = MockPredictor()
        mock_features = MockFeatures()
        
        # Create backtester and run
        backtest = BackTest(mock_predictor, mock_features)
        
        # Mock results for backtest
        backtest.results = create_mock_results()
        
        # Mock the _calculate_metrics method
        backtest._calculate_metrics = mocker.MagicMock(
            return_value={
                'roi': 0.25,
                'total_return': 2500.0,
                'annual_return': 0.30,
                'sharpe_ratio': 1.5,
                'sortino_ratio': 2.0,
                'max_drawdown': -0.1,
                'total_trades': 30,
                'win_rate': 0.6,
                'max_consecutive_wins': 5,
                'max_consecutive_losses': 2,
                'total_fees': 150.0,
                'profit_factor': 1.8,
                'f1_score': 0.75
            }
        )
        
        # Mock the _create_performance_figure method
        backtest._create_performance_figure = mocker.MagicMock()
        
        # Mock the _create_prediction_figure method
        backtest._create_prediction_figure = mocker.MagicMock()
        
        # Mock the _create_trade_history_table method
        backtest._create_trade_history_table = mocker.MagicMock()
        
        # Call dashboard method
        app = backtest.dashboard(debug=False)
        
        # Assertions
        assert app is mock_app
        mock_dash.assert_called_once()
        backtest._calculate_metrics.assert_called_once()
        backtest._create_performance_figure.assert_called_once()
        backtest._create_prediction_figure.assert_called_once()
        backtest._create_trade_history_table.assert_called_once()

if __name__ == '__main__':
    unittest.main() 
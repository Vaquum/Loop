"""
Test script for the extract_features function in experiment.py.

This script tests that the extract_features function in experiment.py
correctly delegates to the Features.extract_features method.
"""

import unittest
import numpy as np
import pandas as pd
import os
import sys

# Add the parent directory to sys.path to import loop modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestExtractFeatures(unittest.TestCase):
    
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

if __name__ == '__main__':
    unittest.main() 
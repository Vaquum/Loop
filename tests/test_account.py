import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime

from loop.account import Account


class TestAccount(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures for each test method"""
        # Initialize with standard starting balance
        self.start_usdt = 1000
        self.account = Account(self.start_usdt)
        
        # Sample timestamp for testing
        self.timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Sample price for testing
        self.price_usdt = 50000.0
        
    def test_init(self):
        """Test initialization of the Account class"""
        # Verify Account initializes correctly
        account = Account(start_usdt=1500)
        
        # Check position_id is initialized to 0
        self.assertEqual(account.position_id, 0)
        
        # Check account dictionary is created properly
        self.assertEqual(account.account['total_usdt'][-1], 1500)
        self.assertEqual(account.account['total_btc'][-1], 0)
        self.assertEqual(account.account['action'][-1], 'hold')
        self.assertEqual(account.account['credit_usdt'][-1], 1500)
        self.assertEqual(account.account['debit_usdt'][-1], 0)
        
    def test_init_account(self):
        """Test the _init_account private method"""
        # Call the private method directly
        account_dict = self.account._init_account(credit_usdt=2000)
        
        # Verify the account dictionary structure
        self.assertEqual(account_dict['position_id'][0], 0)
        self.assertEqual(account_dict['action'][0], 'hold')
        self.assertEqual(account_dict['credit_usdt'][0], 2000)
        self.assertEqual(account_dict['debit_usdt'][0], 0)
        self.assertEqual(account_dict['total_usdt'][0], 2000)
        self.assertEqual(account_dict['total_btc'][0], 0)
        self.assertEqual(account_dict['amount_borrowed_btc'][0], 0)
        self.assertEqual(account_dict['amount_covered_btc'][0], 0)
        
    def test_update_id(self):
        """Test the update_id method"""
        initial_id = self.account.position_id
        self.account.update_id()
        self.assertEqual(self.account.position_id, initial_id + 1)
        
        # Update multiple times
        self.account.update_id()
        self.account.update_id()
        self.assertEqual(self.account.position_id, initial_id + 3)
        
    def test_update_account_buy(self):
        """Test update_account with 'buy' action"""
        # Calculate expected values
        amount = 500  # USDT to spend
        expected_btc_bought = round(amount / self.price_usdt, 7)  # BTC received
        
        # Call update_account with 'buy'
        self.account.update_id()
        self.account.update_account(
            timestamp=self.timestamp,
            action='buy',
            amount=amount,
            price_usdt=self.price_usdt
        )
        
        # Verify account is updated correctly
        self.assertEqual(self.account.account['position_id'][-1], 1)
        self.assertEqual(self.account.account['action'][-1], 'buy')
        self.assertEqual(self.account.account['timestamp'][-1], self.timestamp)
        self.assertEqual(self.account.account['credit_usdt'][-1], 0)
        self.assertEqual(self.account.account['debit_usdt'][-1], amount)
        self.assertEqual(self.account.account['amount_bought_btc'][-1], expected_btc_bought)
        self.assertEqual(self.account.account['amount_sold_btc'][-1], 0)
        self.assertEqual(self.account.account['buy_price_usdt'][-1], self.price_usdt)
        self.assertEqual(self.account.account['sell_price_usdt'][-1], 0)
        self.assertEqual(self.account.account['total_usdt'][-1], self.start_usdt - amount)
        self.assertEqual(self.account.account['total_btc'][-1], expected_btc_bought)
        
    def test_update_account_sell(self):
        """Test update_account with 'sell' action"""
        # First buy some BTC to sell later
        buy_amount = 500  # USDT to spend
        btc_bought = round(buy_amount / self.price_usdt, 7)  # BTC received
        
        self.account.update_id()
        self.account.update_account(
            timestamp=self.timestamp,
            action='buy',
            amount=buy_amount,
            price_usdt=self.price_usdt
        )
        
        # Now sell the BTC
        sell_amount = btc_bought * self.price_usdt  # USDT to receive
        
        self.account.update_id()
        self.account.update_account(
            timestamp=self.timestamp,
            action='sell',
            amount=sell_amount,
            price_usdt=self.price_usdt
        )
        
        # Verify the sell was recorded correctly
        self.assertEqual(self.account.account['position_id'][-1], 2)
        self.assertEqual(self.account.account['action'][-1], 'sell')
        self.assertEqual(self.account.account['credit_usdt'][-1], sell_amount)
        self.assertEqual(self.account.account['debit_usdt'][-1], 0)
        self.assertEqual(self.account.account['amount_bought_btc'][-1], 0)
        self.assertEqual(self.account.account['amount_sold_btc'][-1], btc_bought)
        self.assertEqual(self.account.account['buy_price_usdt'][-1], 0)
        self.assertEqual(self.account.account['sell_price_usdt'][-1], self.price_usdt)
        self.assertEqual(self.account.account['total_usdt'][-1], self.start_usdt)  # Back to starting balance
        self.assertEqual(self.account.account['total_btc'][-1], 0)  # No BTC left
        
    def test_update_account_short(self):
        """Test update_account with 'short' action"""
        # Execute a short position
        short_amount = 500  # USDT to short
        expected_btc_borrowed = round(short_amount / self.price_usdt, 7)
        
        self.account.update_id()
        self.account.update_account(
            timestamp=self.timestamp,
            action='short',
            amount=short_amount,
            price_usdt=self.price_usdt
        )
        
        # Verify short position is recorded correctly
        self.assertEqual(self.account.account['position_id'][-1], 1)
        self.assertEqual(self.account.account['action'][-1], 'short')
        self.assertEqual(self.account.account['credit_usdt'][-1], 0)
        self.assertEqual(self.account.account['debit_usdt'][-1], short_amount)
        self.assertEqual(self.account.account['amount_bought_btc'][-1], 0)
        self.assertEqual(self.account.account['amount_sold_btc'][-1], 0)
        self.assertEqual(self.account.account['amount_borrowed_btc'][-1], expected_btc_borrowed)
        self.assertEqual(self.account.account['amount_covered_btc'][-1], 0)
        self.assertEqual(self.account.account['buy_price_usdt'][-1], self.price_usdt)
        self.assertEqual(self.account.account['sell_price_usdt'][-1], 0)
        self.assertEqual(self.account.account['total_usdt'][-1], self.start_usdt - short_amount)
        
    def test_update_account_cover(self):
        """Test update_account with 'cover' action"""
        # First create a short position
        short_amount = 500  # USDT to short
        expected_btc_borrowed = round(short_amount / self.price_usdt, 7)
        
        self.account.update_id()
        self.account.update_account(
            timestamp=self.timestamp,
            action='short',
            amount=short_amount,
            price_usdt=self.price_usdt
        )
        
        # Now cover at a lower price (profit scenario)
        lower_price = self.price_usdt * 0.9  # 10% lower
        cover_amount = expected_btc_borrowed * lower_price
        
        self.account.update_id()
        self.account.update_account(
            timestamp=self.timestamp,
            action='cover',
            amount=cover_amount,
            price_usdt=lower_price
        )
        
        # Calculate expected profit (based on formula in account.py)
        # Formula is: self.account['buy_price_usdt'][-1] + (self.account['buy_price_usdt'][-1] - price_usdt)
        expected_credit = self.price_usdt + (self.price_usdt - lower_price)
        
        # Verify cover is recorded correctly
        self.assertEqual(self.account.account['position_id'][-1], 2)
        self.assertEqual(self.account.account['action'][-1], 'cover')
        self.assertEqual(self.account.account['amount_covered_btc'][-1], expected_btc_borrowed)
        self.assertEqual(self.account.account['sell_price_usdt'][-1], lower_price)
        self.assertEqual(self.account.account['credit_usdt'][-1], expected_credit)
        
        # Check that we made a profit (total_usdt is higher than start)
        self.assertGreater(self.account.account['total_usdt'][-1], self.start_usdt - short_amount + short_amount)  
        
    def test_update_account_hold(self):
        """Test update_account with 'hold' action"""
        self.account.update_id()
        self.account.update_account(
            timestamp=self.timestamp,
            action='hold',
            amount=0,
            price_usdt=self.price_usdt
        )
        
        # Verify hold is recorded correctly (all values should remain unchanged)
        self.assertEqual(self.account.account['position_id'][-1], 1)
        self.assertEqual(self.account.account['action'][-1], 'hold')
        self.assertEqual(self.account.account['credit_usdt'][-1], 0)
        self.assertEqual(self.account.account['debit_usdt'][-1], 0)
        self.assertEqual(self.account.account['amount_bought_btc'][-1], 0)
        self.assertEqual(self.account.account['amount_sold_btc'][-1], 0)
        self.assertEqual(self.account.account['amount_borrowed_btc'][-1], 0)
        self.assertEqual(self.account.account['amount_covered_btc'][-1], 0)
        self.assertEqual(self.account.account['buy_price_usdt'][-1], 0)
        self.assertEqual(self.account.account['sell_price_usdt'][-1], 0)
        self.assertEqual(self.account.account['total_usdt'][-1], self.start_usdt)
        self.assertEqual(self.account.account['total_btc'][-1], 0)
        
    def test_invalid_action(self):
        """Test error raised for invalid action"""
        with self.assertRaises(AssertionError):
            self.account.update_account(
                timestamp=self.timestamp,
                action='invalid_action',
                amount=100,
                price_usdt=self.price_usdt
            )
            
    def test_negative_amount(self):
        """Test error raised for negative amount"""
        with self.assertRaises(AssertionError):
            self.account.update_account(
                timestamp=self.timestamp,
                action='buy',
                amount=-100,
                price_usdt=self.price_usdt
            )
            
    def test_negative_price(self):
        """Test error raised for negative or zero price"""
        with self.assertRaises(AssertionError):
            self.account.update_account(
                timestamp=self.timestamp,
                action='buy',
                amount=100,
                price_usdt=0
            )
            
        with self.assertRaises(AssertionError):
            self.account.update_account(
                timestamp=self.timestamp,
                action='buy',
                amount=100,
                price_usdt=-10
            )
            
    def test_buy_insufficient_funds(self):
        """Test error raised when trying to buy with insufficient funds"""
        with self.assertRaises(ValueError):
            self.account.update_account(
                timestamp=self.timestamp,
                action='buy',
                amount=self.start_usdt + 100,  # More than available
                price_usdt=self.price_usdt
            )
            
    def test_sell_insufficient_btc(self):
        """Test error raised when trying to sell more BTC than owned"""
        with self.assertRaises(ValueError):
            self.account.update_account(
                timestamp=self.timestamp,
                action='sell',
                amount=100,  # No BTC owned yet
                price_usdt=self.price_usdt
            )
            
    def test_short_insufficient_funds(self):
        """Test error raised when trying to short with insufficient funds"""
        with self.assertRaises(ValueError):
            self.account.update_account(
                timestamp=self.timestamp,
                action='short',
                amount=self.start_usdt + 100,  # More than available
                price_usdt=self.price_usdt
            )
            
    def test_cover_mismatch_amount(self):
        """Test error raised when trying to cover with incorrect amount"""
        # First create a short position
        short_amount = 500  # USDT to short
        
        self.account.update_id()
        self.account.update_account(
            timestamp=self.timestamp,
            action='short',
            amount=short_amount,
            price_usdt=self.price_usdt
        )
        
        # Try to cover with incorrect amount
        lower_price = self.price_usdt * 0.9  # 10% lower
        incorrect_cover_amount = short_amount * 0.8  # Not matching borrowed amount
        
        with self.assertRaises(ValueError):
            self.account.update_account(
                timestamp=self.timestamp,
                action='cover',
                amount=incorrect_cover_amount,
                price_usdt=lower_price
            )


if __name__ == '__main__':
    unittest.main() 
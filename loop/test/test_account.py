import unittest
from loop.account import Account

class TestAccountClass(unittest.TestCase):
    def setUp(self):
        # This runs before each test
        self.starting_usdt = 1000
        self.account = Account(start_usdt=self.starting_usdt, testing_mode=True)
    
    def test_initialization(self):
        """Test that the account initializes correctly"""
        # Check initial values
        self.assertEqual(self.account.id, 1)
        self.assertEqual(self.account.account['total_usdt'][0], self.starting_usdt)
        self.assertEqual(self.account.account['total_btc'][0], 0)
        self.assertEqual(self.account.account['position_type'][0], 'none')
        self.assertEqual(self.account.account['timestamp'][0], 'INIT')
    
    def test_buy_action(self):
        """Test the buy action"""
        # Buy BTC worth 500 USDT at 50000 USDT per BTC
        self.account.update_account('buy', 500, 50000)
        
        # Check account state after buy
        self.assertEqual(self.account.account['action'][-1], 'buy')
        self.assertEqual(self.account.account['position_type'][-1], 'long')
        self.assertEqual(self.account.account['debit_usdt'][-1], 500)
        self.assertEqual(self.account.account['amount_bought_btc'][-1], 0.01)  # 500/50000 = 0.01
        self.assertEqual(self.account.account['total_usdt'][-1], 500)  # 1000 - 500 = 500
        self.assertEqual(self.account.account['total_btc'][-1], 0.01)  # Bought 0.01 BTC
        
    def test_sell_action(self):
        """Test the sell action after buying"""
        # First buy some BTC
        self.account.update_account('buy', 500, 50000)
        
        # Now sell 0.005 BTC at 55000 USDT per BTC
        self.account.update_account('sell', 0.005, 55000)
        
        # Check account state after sell
        self.assertEqual(self.account.account['action'][-1], 'sell')
        self.assertEqual(self.account.account['position_type'][-1], 'long')  # Still have 0.005 BTC
        self.assertEqual(self.account.account['credit_usdt'][-1], 275)  # 0.005 * 55000 = 275
        self.assertEqual(self.account.account['amount_sold_btc'][-1], 0.005)
        self.assertEqual(self.account.account['total_usdt'][-1], 775)  # 500 + 275 = 775
        self.assertEqual(self.account.account['total_btc'][-1], 0.005)  # 0.01 - 0.005 = 0.005
        
    def test_sell_all(self):
        """Test selling all BTC"""
        # First buy some BTC
        self.account.update_account('buy', 500, 50000)
        
        # Now sell all BTC (0.01) at 55000 USDT per BTC
        self.account.update_account('sell', 0.01, 55000)
        
        # Check position_type is none after selling all
        self.assertEqual(self.account.account['position_type'][-1], 'none')
        self.assertEqual(self.account.account['total_btc'][-1], 0)
        
    def test_short_action(self):
        """Test the short action"""
        # Short BTC worth 500 USDT at 50000 USDT per BTC
        self.account.update_account('short', 500, 50000)
        
        # Check account state after short
        self.assertEqual(self.account.account['action'][-1], 'short')
        self.assertEqual(self.account.account['position_type'][-1], 'short')
        self.assertEqual(self.account.account['credit_usdt'][-1], 500)
        self.assertEqual(self.account.account['borrowed_btc'][-1], 0.01)  # 500/50000 = 0.01
        self.assertEqual(self.account.account['total_usdt'][-1], 1500)  # 1000 + 500 = 1500
        self.assertEqual(self.account.account['total_borrowed_btc'][-1], 0.01)
        
    def test_cover_action(self):
        """Test the cover action after shorting"""
        # First short some BTC
        self.account.update_account('short', 500, 50000)
        
        # Now cover 0.005 BTC at 45000 USDT per BTC
        self.account.update_account('cover', 0.005, 45000)
        
        # Check account state after cover
        self.assertEqual(self.account.account['action'][-1], 'cover')
        self.assertEqual(self.account.account['position_type'][-1], 'short')  # Still have 0.005 BTC borrowed
        self.assertEqual(self.account.account['debit_usdt'][-1], 225)  # 0.005 * 45000 = 225
        self.assertEqual(self.account.account['covered_btc'][-1], 0.005)
        self.assertEqual(self.account.account['total_usdt'][-1], 1275)  # 1500 - 225 = 1275
        self.assertEqual(self.account.account['total_borrowed_btc'][-1], 0.005)  # 0.01 - 0.005 = 0.005
        
    def test_cover_all(self):
        """Test covering all borrowed BTC"""
        # First short some BTC
        self.account.update_account('short', 500, 50000)
        
        # Now cover all borrowed BTC (0.01) at 45000 USDT per BTC
        self.account.update_account('cover', 0.01, 45000)
        
        # Check position_type is none after covering all
        self.assertEqual(self.account.account['position_type'][-1], 'none')
        self.assertEqual(self.account.account['total_borrowed_btc'][-1], 0)
    
    def test_insufficient_funds_buy(self):
        """Test buying with insufficient funds"""
        # Try to buy BTC worth 1500 USDT with only 1000 USDT balance
        self.account.update_account('buy', 1500, 50000)
        
        # Should only use available balance
        self.assertEqual(self.account.account['debit_usdt'][-1], 1000)
        self.assertEqual(self.account.account['amount_bought_btc'][-1], 0.02)  # 1000/50000 = 0.02
        self.assertEqual(self.account.account['total_usdt'][-1], 0)  # Used all USDT
        
    def test_insufficient_btc_sell(self):
        """Test selling more BTC than available"""
        # First buy some BTC
        self.account.update_account('buy', 500, 50000)  # Buys 0.01 BTC
        
        # Try to sell 0.02 BTC (more than the 0.01 available)
        self.account.update_account('sell', 0.02, 55000)
        
        # Should only sell available BTC
        self.assertEqual(self.account.account['amount_sold_btc'][-1], 0.01)  # Only had 0.01 BTC
        self.assertEqual(self.account.account['credit_usdt'][-1], 550)  # 0.01 * 55000 = 550
        self.assertEqual(self.account.account['total_btc'][-1], 0)  # Sold all BTC
        
    def test_insufficient_funds_cover(self):
        """Test covering with insufficient funds"""
        # First short some BTC
        self.account.update_account('short', 1000, 50000)  # Shorts 0.02 BTC, balance now 2000 USDT
        
        # Spend most of the balance
        self.account.update_account('buy', 1900, 50000)  # Spends 1900 USDT, balance now 100 USDT
        
        # Try to cover all borrowed BTC (0.02) at 10000 USDT per BTC (would need 200 USDT)
        self.account.update_account('cover', 0.02, 10000)
        
        # Should only cover what we can afford
        self.assertEqual(self.account.account['covered_btc'][-1], 0.01)  # Can only cover 100/10000 = 0.01 BTC
        self.assertEqual(self.account.account['debit_usdt'][-1], 100)  # Used all remaining USDT
        self.assertEqual(self.account.account['total_borrowed_btc'][-1], 0.01)  # Still have 0.01 BTC borrowed
        
    def test_complete_long_cycle(self):
        """Test a complete long position cycle"""
        # Buy BTC
        self.account.update_account('buy', 500, 50000)  # Buy 0.01 BTC
        
        # Sell half at profit
        self.account.update_account('sell', 0.005, 60000)  # Sell 0.005 BTC
        
        # Sell remaining at loss
        self.account.update_account('sell', 0.005, 40000)  # Sell 0.005 BTC
        
        # Check final state
        self.assertEqual(self.account.account['position_type'][-1], 'none')
        self.assertEqual(self.account.account['total_btc'][-1], 0)
        expected_usdt = 1000 - 500 + 0.005*60000 + 0.005*40000
        self.assertEqual(self.account.account['total_usdt'][-1], expected_usdt)
        
    def test_complete_short_cycle(self):
        """Test a complete short position cycle"""
        # Short BTC
        self.account.update_account('short', 500, 50000)  # Short 0.01 BTC
        
        # Cover half at profit
        self.account.update_account('cover', 0.005, 40000)  # Cover 0.005 BTC
        
        # Cover remaining at loss
        self.account.update_account('cover', 0.005, 60000)  # Cover 0.005 BTC
        
        # Check final state
        self.assertEqual(self.account.account['position_type'][-1], 'none')
        self.assertEqual(self.account.account['total_borrowed_btc'][-1], 0)
        expected_usdt = 1000 + 500 - 0.005*40000 - 0.005*60000
        self.assertEqual(self.account.account['total_usdt'][-1], expected_usdt)
        
    def test_hold_action(self):
        """Test the hold action"""
        initial_usdt = self.account.account['total_usdt'][-1]
        initial_btc = self.account.account['total_btc'][-1]
        
        # Execute hold action
        self.account.update_account('hold', 0, 50000)
        
        # Check that balances don't change
        self.assertEqual(self.account.account['action'][-1], 'hold')
        self.assertEqual(self.account.account['total_usdt'][-1], initial_usdt)
        self.assertEqual(self.account.account['total_btc'][-1], initial_btc)
    
    def test_zero_price(self):
        """Test operations with zero price"""
        # Try to buy BTC with zero price (should prevent division by zero)
        self.account.update_account('buy', 100, 0)
        
        # Check that amount_bought_btc is nonzero (should use min price)
        self.assertGreater(self.account.account['amount_bought_btc'][-1], 0)
        self.assertEqual(self.account.account['debit_usdt'][-1], 100)
        
    def test_empty_account(self):
        """Test operations with no balance"""
        # Create empty account
        empty_account = Account(start_usdt=0, testing_mode=True)
        
        # Try to buy BTC
        empty_account.update_account('buy', 100, 50000)
        
        # Check that nothing changes
        self.assertEqual(empty_account.account['debit_usdt'][-1], 0)
        self.assertEqual(empty_account.account['total_usdt'][-1], 0)
        
    def test_sequential_operations(self):
        """Test multiple sequential operations"""
        # Series of operations
        operations = [
            ('buy', 300, 50000),    # Buy 0.006 BTC for 300 USDT
            ('buy', 200, 52000),    # Buy 0.0038 BTC for 200 USDT
            ('sell', 0.004, 55000), # Sell 0.004 BTC for 220 USDT
            ('short', 250, 54000),  # Short 0.0046 BTC for 250 USDT
            ('cover', 0.002, 52000), # Cover 0.002 BTC for 104 USDT
            ('hold', 0, 53000),     # Hold
            ('sell', 0.0058, 56000), # Sell all remaining BTC
            ('cover', 0.0026, 51000) # Cover all remaining borrowed BTC
        ]
        
        # Execute operations
        for action, amount, price in operations:
            self.account.update_account(action, amount, price)
        
        # Check final state
        self.assertEqual(self.account.account['position_type'][-1], 'none')
        self.assertEqual(self.account.account['total_btc'][-1], 0)
        self.assertEqual(self.account.account['total_borrowed_btc'][-1], 0)
        
        # Calculate expected final USDT balance manually
        # Starting with 1000 USDT
        # - 300 for first buy
        # - 200 for second buy
        # + 220 for first sell (0.004 * 55000)
        # + 250 for short
        # - 104 for first cover (0.002 * 52000)
        # + 0 for hold
        # + 324.8 for second sell (0.0058 * 56000)
        # - 132.6 for second cover (0.0026 * 51000)
        expected_usdt = 1000 - 300 - 200 + 220 + 250 - 104 + 0 + 324.8 - 132.6
        self.assertAlmostEqual(self.account.account['total_usdt'][-1], expected_usdt, places=2)
        
    def test_mixed_long_short(self):
        """Test having both long and short positions simultaneously"""
        # Buy some BTC
        self.account.update_account('buy', 300, 50000)  # Buy 0.006 BTC
        
        # Short some BTC
        self.account.update_account('short', 250, 55000)  # Short 0.0045... BTC
        
        # Check state - should be possible to have both positions
        self.assertEqual(self.account.account['position_type'][-1], 'mixed')
        self.assertAlmostEqual(self.account.account['total_btc'][-1], 0.006, places=4)
        self.assertAlmostEqual(self.account.account['total_borrowed_btc'][-1], 0.0045, places=4)
        
        # Sell some of the long position
        self.account.update_account('sell', 0.003, 54000)
        
        # Cover some of the short position
        self.account.update_account('cover', 0.002, 52000)
        
        # Check final state
        self.assertAlmostEqual(self.account.account['total_btc'][-1], 0.003, places=4)  # 0.006 - 0.003
        self.assertAlmostEqual(self.account.account['total_borrowed_btc'][-1], 0.0025, places=4)  # 0.0045... - 0.002
        self.assertEqual(self.account.account['position_type'][-1], 'mixed')  # Still have both positions
        
    def test_negative_amount(self):
        """Test operations with negative amount (should be treated as 0)"""
        # Try to buy negative BTC
        self.account.update_account('buy', -100, 50000)
        
        # Check that nothing changes in balance
        self.assertEqual(self.account.account['debit_usdt'][-1], 0)
        self.assertEqual(self.account.account['amount_bought_btc'][-1], 0)
        self.assertEqual(self.account.account['total_usdt'][-1], self.starting_usdt)


if __name__ == '__main__':
    unittest.main()
    print("All tests passed!")
from datetime import datetime

class Account:
        
    '''Account class is used to keep track of account information for both long and short positions'''
    
    def __init__(self, start_usdt):
        '''Initializes the account object.
        
        start_usdt | int | starting usdt balance
        '''
        self.position_id = 0
        
        self.account = self._init_account(credit_usdt=start_usdt)
        
    def _init_account(self, credit_usdt):
        
        '''Initializes the account with the starting balance.
        
        credit_usdt | int | starting usdt balance
        '''
        account = {'position_id': [self.position_id],
                   'action': ['hold'],
                   'timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                   'credit_usdt': [credit_usdt],
                   'debit_usdt': [0],
                   'amount_bought_btc': [0],
                   'amount_sold_btc': [0],
                   'amount_borrowed_btc': [0],        # Added field: tracks borrowed BTC for shorts
                   'amount_covered_btc': [0],         # Added field: tracks covered (repaid) BTC
                   'buy_price_usdt': [0],
                   'sell_price_usdt': [0],
                   'total_usdt': [credit_usdt],
                   'total_btc': [0]}
        
        return account
    
    def update_account(self,
                      timestamp,
                      action,
                      amount,
                      price_usdt):
        
        '''Updates the account information based on the action taken.
        
        action | str | 'buy', 'sell', 'short', 'cover', or 'hold'
        timestamp | datetime | current timestamp
        amount | int | amount in USDT (for buy/short) or BTC (for sell/cover)
        price_usdt | float | current price of the asset
        '''
        
        # Initialize values
        credit_usdt = 0
        debit_usdt = 0
        amount_bought_btc = 0
        amount_sold_btc = 0
        amount_borrowed_btc = 0
        amount_covered_btc = 0
        buy_price_usdt = 0
        sell_price_usdt = 0

        assert action in ['buy', 'sell', 'short', 'cover', 'hold'], "ERROR: " + action + " not suported."
        assert amount >= 0, "ERROR: amount can't be negative."
        assert price_usdt > 0, "ERROR: price_usdt has to be positive."
        
        if action == 'buy':

            if amount > self.account['total_usdt'][-1]:
                raise ValueError("ERROR: amount can't be larger than total_usdt.")
                
            debit_usdt = amount
            amount_bought_btc = round(debit_usdt / price_usdt, 7)
            buy_price_usdt = price_usdt
        
        elif action == 'sell':

            if (amount / price_usdt) > self.account['total_btc'][-1]:
                raise ValueError("ERROR: amount / price_usdt can't be more than total_btc")

            credit_usdt = amount
            amount_sold_btc = round(credit_usdt / price_usdt, 7)
            sell_price_usdt = price_usdt
            
        elif action == 'short':

            if amount > self.account['total_usdt'][-1]:
                raise ValueError("ERROR: amount can't be larger than total_usdt.")

            amount_borrowed_btc = round(amount / price_usdt, 7)
            debit_usdt = amount
            buy_price_usdt = price_usdt
            
        elif action == 'cover':

            if (amount / price_usdt) != self.account['amount_borrowed_btc'][-1]:
                raise ValueError("ERROR: To cover, amount / price_usdt has to equal amount_borrowed_btc")

            amount_covered_btc = (amount / price_usdt)
            credit_usdt = self.account['buy_price_usdt'][-1] + (self.account['buy_price_usdt'][-1] - price_usdt)
            sell_price_usdt = price_usdt

        # Update the account
        self.account['position_id'].append(self.position_id)
        self.account['action'].append(action)
        self.account['timestamp'].append(timestamp)
        self.account['credit_usdt'].append(credit_usdt)
        self.account['debit_usdt'].append(debit_usdt)
        self.account['amount_bought_btc'].append(amount_bought_btc)
        self.account['amount_sold_btc'].append(amount_sold_btc)
        self.account['amount_borrowed_btc'].append(amount_borrowed_btc)
        self.account['amount_covered_btc'].append(amount_covered_btc)
        self.account['buy_price_usdt'].append(buy_price_usdt)
        self.account['sell_price_usdt'].append(sell_price_usdt)
    
        # Calculate totals
        total_btc = sum(self.account['amount_bought_btc']) - sum(self.account['amount_sold_btc'])
        total_usdt = sum(self.account['credit_usdt']) - sum(self.account['debit_usdt'])
        
        self.account['total_btc'].append(round(total_btc, 7))        
        self.account['total_usdt'].append(round(total_usdt, 2))
        
    def update_id(self):
        
        '''Will increment id by one. This has to be run always before
        updating account or book.'''
        
        self.position_id += 1

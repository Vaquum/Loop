from datetime import datetime

class Account:
    '''Account class is used to keep track of account information for both long and short positions'''
    
    def __init__(self, start_usdt):
        '''Initializes the account object.
        
        start_usdt | int | starting usdt balance
        '''
        self.id = 0
        
        self.account = self._init_account(credit_usdt=start_usdt)
        
        self.update_id()
        
    def _init_account(self, credit_usdt):
        '''Initializes the account with the starting balance.
        
        credit_usdt | int | starting usdt balance
        '''
        account = {'id': [self.id],
                   'action': ['hold'],
                   'position_type': ['none'],  # Added field: none, long, or short
                   'timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                   'credit_usdt': [credit_usdt],
                   'debit_usdt': [0],
                   'amount_bought_btc': [0],
                   'amount_sold_btc': [0],
                   'borrowed_btc': [0],        # Added field: tracks borrowed BTC for shorts
                   'covered_btc': [0],         # Added field: tracks covered (repaid) BTC
                   'buy_price_usdt': [0],
                   'sell_price_usdt': [0],
                   'total_usdt': [credit_usdt],
                   'total_btc': [0],
                   'total_borrowed_btc': [0]}  # Running total of borrowed BTC
        
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
        borrowed_btc = 0
        covered_btc = 0
        buy_price_usdt = 0
        sell_price_usdt = 0
        position_type = self.account['position_type'][-1]  # Default to previous position
        
        if action == 'buy':

            if amount > self.account['total_usdt'][-1]:
                debit_usdt = self.account['total_usdt'][-1]

            else:
                debit_usdt = amount
            
            amount_bought_btc = round(debit_usdt / price_usdt, 4)
            buy_price_usdt = price_usdt
            position_type = 'long'
        
        elif action == 'sell':

            if self.account['total_btc'][-1] <= 0:
                credit_usdt = 0
                amount_sold_btc = 0
            
            elif amount > self.account['total_btc'][-1]:
                amount_sold_btc = self.account['total_btc'][-1]
                credit_usdt = amount_sold_btc * price_usdt
            
            else:
                amount_sold_btc = amount
                credit_usdt = amount * price_usdt
            
            sell_price_usdt = price_usdt
            if self.account['total_btc'][-1] - amount_sold_btc <= 0:
                position_type = 'none'
            
        elif action == 'short':
            
            # Borrow BTC and sell it for USDT (going short)
            borrowed_btc = round(amount / price_usdt, 4)
            amount_sold_btc = borrowed_btc
            credit_usdt = amount
            sell_price_usdt = price_usdt
            position_type = 'short'
            
        elif action == 'cover':
            
            # Buy BTC to cover short position (exiting short)
            if self.account['total_borrowed_btc'][-1] <= 0:
                debit_usdt = 0
                covered_btc = 0
            elif amount > self.account['total_borrowed_btc'][-1]:
                covered_btc = self.account['total_borrowed_btc'][-1]
                debit_usdt = covered_btc * price_usdt
            else:
                covered_btc = amount
                debit_usdt = amount * price_usdt
            
            if debit_usdt > self.account['total_usdt'][-1]:
                debit_usdt = self.account['total_usdt'][-1]
                covered_btc = round(debit_usdt / price_usdt, 4)
            
            amount_bought_btc = covered_btc
            buy_price_usdt = price_usdt
            
            if self.account['total_borrowed_btc'][-1] - covered_btc <= 0:
                position_type = 'none'
            
        elif action == 'hold':
            # No change in position
            pass
            
        # Update the account
        self.account['id'].append(self.id)
        self.account['action'].append(action)
        self.account['position_type'].append(position_type)
        self.account['timestamp'].append(timestamp)
        self.account['credit_usdt'].append(credit_usdt)
        self.account['debit_usdt'].append(debit_usdt)
        self.account['amount_bought_btc'].append(amount_bought_btc)
        self.account['amount_sold_btc'].append(amount_sold_btc)
        self.account['borrowed_btc'].append(borrowed_btc)
        self.account['covered_btc'].append(covered_btc)
        self.account['buy_price_usdt'].append(buy_price_usdt)
        self.account['sell_price_usdt'].append(sell_price_usdt)
    
        # Calculate totals
        total_btc = sum(self.account['amount_bought_btc']) - sum(self.account['amount_sold_btc'])
        total_usdt = sum(self.account['credit_usdt']) - sum(self.account['debit_usdt'])
        total_borrowed_btc = sum(self.account['borrowed_btc']) - sum(self.account['covered_btc'])
        
        self.account['total_btc'].append(round(total_btc, 4))        
        self.account['total_usdt'].append(round(total_usdt, 2))
        self.account['total_borrowed_btc'].append(round(total_borrowed_btc, 4))
        
    def update_id(self):
        '''Will increment id by one. This has to be run always before
        updating account or book.'''
        
        self.id += 1
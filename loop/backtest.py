import pandas as pd

from loop.account import Account

class Backtest:

    def __init__(self) -> None:

        pass

    def run(self, start_usdt: int, predictor, prep, data: pd.DataFrame) -> None:

        x_live = prep(data, mode='predict')
        signals = pd.cut(predictor(x_live), 3, labels=['short', 'hold', 'buy'])
        account = Account(start_usdt=start_usdt)

        for day_idx in range(len(data)):

            timestamp = data.iloc[day_idx]['open_time']
            day_open = data.iloc[day_idx]['open']
            day_close = data.iloc[day_idx]['close']
            total_usdt = account.account['total_usdt'][-1]
            
            account.update_account(timestamp, signals[day_idx], total_usdt, day_open)
            account.update_id()

            if signals[day_idx] == 'buy':
                account.update_account(timestamp, 'sell', total_usdt, day_close)
                account.update_id()
            
            elif signals[day_idx] == 'short':
                account.update_account(timestamp, 'cover', total_usdt, day_close)
                account.update_id()

        return account.account
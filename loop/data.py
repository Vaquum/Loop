import pandas as pd
from typing import Any, Dict, List
from loop.utils.get_klines_historical import get_klines_historical

class HistoricalKlinesData:
    
    data: pd.DataFrame
    input_data: Dict[str, pd.DataFrame]

    
    def __init__(self,
                 data: pd.DataFrame = None,
                 data_file_path: str = None,
                 data_start_date: str = None, # YYYY-MM-DD
                 data_end_date: str = None, # YYYY-MM-DD
                 data_interval: str = None, # '1h' or '1d'
                 drop_na: bool = True) -> None:

        '''Get historical klines data from a DataFrame, file, or API.

        If data is DataFrame, then data_file_path, data_start_date,
        data_end_date, and data_interval must be None. 

        If data_file_path is string, then data, data_start_date,
        data_end_date, and data_interval must be None.

        data_start_date, data_end_date, and data_interval must all
        be set together, at which time data and data_file_path
        must be None
        
        '''

        if isinstance(data, pd.DataFrame):
            self.data = data  
        
        elif isinstance(data_file_path, str):
            self.data = pd.read_csv(data_file_path)

        elif isinstance(data_start_date, str):
            if isinstance(data_end_date, str):
                if isinstance(data_interval, str):
                    self.data = get_klines_historical(data_interval,
                                                      data_start_date,
                                                      data_end_date)

        else: 
            raise ValueError("Invalid data input")
        
        if drop_na:
            self.data = self.data.dropna()

        # All the int columns
        self.int_cols = ['open_time', 'close_time', 'num_trades']

        # All the float columns
        self.float_cols = ['open', 'high', 'low', 'close', 'volume', 
                           'qav', 'taker_base_vol', 'taker_quote_vol', 'ignore']

        # All the datetime columns
        self.data['open_time'] = pd.to_datetime(self.data['open_time'])

        # Ensure that column lists and self.data have the same columns
        all_cols = self.int_cols + self.float_cols + ['open_time']
        assert set(self.data.columns) == set(all_cols), 'Input data columns do not match the expectation.'
        
        self.input_data = {}
    
    def split_data(self,
                   train_ratio: int = 1,
                   test_ratio: int = 1,
                   validate_ratio: int = 1) -> 'HistoricalKlinesData':

        if len(self.data) == 0:
            raise ValueError("Cannot split an empty dataset")
            
        if not all(isinstance(x, int) and x >= 0 for x in [train_ratio, test_ratio, validate_ratio]):
            raise ValueError("All ratios must be non-negative integers")
        
        total_ratio = train_ratio + test_ratio + validate_ratio
        
        if total_ratio == 0:
            raise ValueError("At least one ratio must be positive")
        
        min_required_rows = sum(1 for ratio in [train_ratio, test_ratio, validate_ratio] if ratio > 0)
        if len(self.data) < min_required_rows:
            raise ValueError(f"Dataset has {len(self.data)} rows but at least {min_required_rows} rows are required for the requested split")
        
        if 'open_time' in self.data.columns and not self.data['open_time'].is_monotonic_increasing:
            raise ValueError("Data is not sorted chronologically by 'open_time'")
        
        self.input_data = {}
        
        n = len(self.data)
        train_end = int(n * (train_ratio / total_ratio))
        test_end = train_end + int(n * (test_ratio / total_ratio))
        
        self.input_data['train'] = self.data.iloc[:train_end].copy() if train_ratio > 0 else pd.DataFrame()
        self.input_data['test'] = self.data.iloc[train_end:test_end].copy() if test_ratio > 0 else pd.DataFrame()
        self.input_data['validate'] = self.data.iloc[test_end:].copy() if validate_ratio > 0 else pd.DataFrame()
        
        return self

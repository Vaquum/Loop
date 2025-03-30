from typing import Any, Dict, List, Optional
import pandas as pd
import ta
from sklearn.preprocessing import RobustScaler, StandardScaler

class Features:
    
    def __init__(self, data: Any, scaler_type: str = 'robust', live_data: bool = False) -> None:
        
        if not hasattr(data, 'input_data') or not data.input_data:
            raise ValueError("Data must be split before features can be created. Call data.split_data() first.")
        
        self.data = data.data

        self.live_data = live_data

        self.input_data = {}
        for split_name, df in data.input_data.items():
            self.input_data[split_name] = df.copy()
        
        self.scaler = None
        self.production_features: Dict[str, pd.DataFrame] = {}
        
        if scaler_type not in ['robust', 'standard']:
            raise ValueError("scaler_type must be 'robust' or 'standard'")
        
        self.scaler_type = scaler_type

    def add_features(self) -> 'Features':

        '''Process raw data into features for the specified datasets.

        Args:
            None
        Returns:
            Features: The current Features instance with processed features.
            
        Raises:
            ValueError: If any required columns are missing from the input data.
            KeyError: If a specified dataset name is not found in input_data.

        '''

        self._add_indicators()
        self._scale_features()

        for split_name in self.production_features.keys():
            if len(self.input_data[split_name]) > 0:
                self.production_features[split_name] = self.production_features[split_name].iloc[1:-1]
                self.production_features[split_name] = self.production_features[split_name].drop(columns=['trend_stc',
                                                                                                        'trend_psar_up',
                                                                                                        'trend_psar_down'])
        return self

    def _add_indicators(self) -> 'Features':

        self._check_required_columns(['close', 'high', 'low', 'volume', 'open'])

        for split_name, df in self.input_data.items():
            if len(self.input_data[split_name]) > 0:

                # Use the whole data to add indicators
                whole_data = ta.add_all_ta_features(df=self.data,
                                                    open='open',
                                                    close='close',
                                                    high='high',
                                                    low='low',
                                                    volume='volume')
                
                # Add the indicators to the split data
                self.input_data[split_name] = whole_data.loc[self.input_data[split_name].index]
                
                # Add ROC
                self.input_data[split_name]['close_roc'] = self.input_data[split_name]['close'].pct_change()
    
        return self
    
    def _scale_features(self) -> 'Features':

        self.production_features = {}
        
        if self.scaler_type == 'robust':
            self.scaler = RobustScaler()

        elif self.scaler_type == 'standard':
            self.scaler = StandardScaler()
        
        train_df = self.input_data['train']
        feature_cols = self._get_feature_columns(train_df)
        
        self.scaler.fit(train_df[feature_cols])
        
        for split_name, df in self.input_data.items():

            scaled_df = df.copy()
            
            if len(df) > 0:

                split_feature_cols = [col for col in df.columns if col in feature_cols]
                
                if split_feature_cols:

                    if self.scaler_type == 'robust':
                        split_scaler = RobustScaler()
            
                    elif self.scaler_type == 'standard':
                        split_scaler = StandardScaler()
                    
                    split_scaler.fit(train_df[split_feature_cols])
                    
                    scaled_df[split_feature_cols] = split_scaler.transform(df[split_feature_cols])
            
            self.production_features[split_name] = scaled_df
        
        return self

    def _get_feature_columns(self, df: pd.DataFrame = None) -> List[str]:

        # Columns to exclude from scaling (timestamps, indices, etc.)
        exclude_cols = [
            'open', 'high', 'low', 'close', 'volume',
            'close_time', 'number_of_trades', 'ignore', 'close_date',
            'qav', 'taker_base_vol', 'taker_quote_vol', 'momentum_stoch_rsi_d',
            'momentum_rsi', 'momentum_stoch_rsi', 'momentum_stoch_rsi_k', 
            'momentum_stoch', 'momentum_stoch_signal', 'momentum_wr',
            'volatility_bbp', 'volatility_bbhi', 'volatility_bbli',
            'volatility_kcp', 'volatility_kchi', 'volatility_kcli',
            'volatility_dcp', 'trend_psar_up_indicator', 'trend_psar_down_indicator',
            'trend_visual_ichimoku_a', 'trend_visual_ichimoku_b']
        
        # Get all numeric columns except those in exclude_cols
        numeric_cols = [
            col for col in df.select_dtypes(include=['int64', 'float64']).columns 
            if col not in exclude_cols]

        return numeric_cols
        
    def _check_required_columns(self, required_cols: List[str]) -> None:

        df = next((df for df in self.input_data.values() if len(df) > 0), None)
        if df is None:
            raise ValueError("All datasets are empty")
            
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Required columns {missing_cols} are missing from data")
    
    def _has_required_columns(self, required_cols: List[str]) -> bool:

        try:
            self._check_required_columns(required_cols)
            return True
        except ValueError:
            return False

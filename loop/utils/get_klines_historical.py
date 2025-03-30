import binancial
import pandas as pd


def get_klines_historical(interval: str, 
                          start_date: str, 
                          end_date: str) -> pd.DataFrame:
    
    from dotenv import load_dotenv
    import os

    load_dotenv()

    api_key = os.getenv("binance_api_key")
    api_secret = os.getenv("binance_api_secret")

    if api_key is None or api_secret is None:
        raise ValueError("API key and/or secret not found in .env file")

    client = binancial.utils.init_binance_api('historical', api_key, api_secret)
    klines = binancial.data.get_klines_historical(client,
                                                  interval=interval,
                                                  start_date=start_date,
                                                  end_date=end_date)
    
    return klines

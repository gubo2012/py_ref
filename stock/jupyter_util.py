from datetime import datetime
import pandas as pd

def get_ticker(symbol):
    # client to synced db
    db_client = SyncDB
    # collection name for daily stock data is stock_daily
    cl_name = 'stock_daily'

    # set the symbol to query
#     symbol = 'QQQ'
    cursor = db_client.find(cl_name, {'symbol': symbol})
    result_list = list(cursor)
    df = pd.DataFrame(result_list)[['date', 'symbol', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
#     df.set_index('date', inplace=True, drop=True)
    df.sort_values('date', inplace=True)
    
    return df


def get_short_vol(symbol):
    db_client = SyncDB
    cl_name = 'MA_FINRA_ShortVolume'
    
    cursor = db_client.find(cl_name, {'Symbol': symbol})
    
    result_list = list(cursor)
    df = pd.DataFrame(result_list)[['_id', 'Symbol', 'ShortVolume', 'ShortExemptVolume', 'TotalVolume',
                                    'Market', 'date', 'Price']]
    df_sum = df.groupby(['date']).agg({'ShortVolume':'sum'})
    
    return df_sum


def get_ticker_merged(symbol):
    df = get_ticker(symbol)
    df_short = get_short_vol(symbol)

    df = pd.merge(df, df_short, on='date', how='left')
    df['ShortVolume'] = df['ShortVolume'].fillna(-1)
    df['short_vol_pct'] = df['ShortVolume'] / df['Volume']
    
    return df
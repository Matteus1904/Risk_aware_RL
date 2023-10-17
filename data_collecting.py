import requests
import pandas as pd
import math
from datetime import datetime
from tqdm import tqdm_notebook

seconds_per_unit = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}

def convert_to_seconds(s):
    return int(s[:-1]) * seconds_per_unit[s[-1]]

def klines(market = 'BTCUSDT', tick_interval = '30m', startDay = "01/10/2023", endDay="03/10/2023"):
    startTime = int(datetime.strptime(startDay,"%d/%m/%Y").timestamp() + 10800 - convert_to_seconds(tick_interval))
    endTime = int(datetime.strptime(endDay,"%d/%m/%Y").timestamp() + 10799)
    interval = convert_to_seconds(tick_interval)
    n = (endTime - startTime)/interval
    for i in tqdm_notebook(range(math.ceil(n/1000))):
        start = startTime +i*interval*1000
        end = min(start+interval*1000, endTime)
        url = 'https://api.binance.com/api/v3/klines?symbol='+market+'&interval='+tick_interval+'&startTime='+ str(start*1000)+'&endTime='+str(end*1000)+'&limit=1000'
        data_one = pd.DataFrame(requests.get(url).json())[[0, 4]]
        data_one.rename(columns={0: 'timestamp', 4: 'close_price'}, inplace=True)
        data_one.sort_values(by=['timestamp'], inplace=True)
        if i==0:
            data = data_one
        else:
            data = pd.concat([data, data_one], ignore_index=True)
    data['timestamp'] = pd.to_datetime(data['timestamp']/1000, unit='s')
    data['close_price'] = pd.to_numeric(data['close_price'])
    data['returns'] = (data.close_price -data.close_price.shift(1))/ data.close_price.shift(1)
    data = data.iloc[1:].reset_index(drop = True)
    data = data[['returns']]
    return data

import requests
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import copy
import time
import random



# from binance import Client, ThreadedWebsocketManager, ThreadedDepthCacheManager


symbol = 'ETHUSDT'
# client = Client(KEY, SECRET)

maxposition = 0.03
stop_percent = 0.01  # 0.01=1%
eth_proffit_array = [[20, 1], [40, 1], [60, 2], [80, 2], [100, 2], [150, 1], [200, 1], [200, 0]]
proffit_array = copy.copy(eth_proffit_array)

pointer = str(random.randint(1000, 9999))


# Get last 500 kandels 5 minutes for Symbol
def get_futures_klines(symbol, limit=100):
    x = requests.get('https://binance.com/fapi/v1/klines?symbol=' + symbol + '&limit=' + str(limit) + '&interval=5m')
    df = pd.DataFrame(x.json())
    df.columns = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'd1', 'd2', 'd3', 'd4', 'd5']
    df = df.drop(['d1', 'd2', 'd3', 'd4', 'd5'], axis=1)
    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float)
    return df


# To find a slope of price line
def ind_slope(series, n):
    array_sl = [j * 0 for j in range(n - 1)]

    for j in range(n, len(series) + 1):
        y = series[j - n:j]
        x = np.array(range(n))
        x_sc = (x - x.min()) / (x.max() - x.min())
        y_sc = (y - y.min()) / (y.max() - y.min())
        x_sc = sm.add_constant(x_sc)
        model = sm.OLS(y_sc, x_sc)
        results = model.fit()
        array_sl.append(results.params[-1])
    slope_angle = (np.rad2deg(np.arctan(np.array(array_sl))))
    return np.array(slope_angle)


# True Range and Average True Range indicator
def ind_ATR(source_DF, n):
    df = source_DF.copy()
    df['H-L'] = abs(df['high'] - df['low'])
    df['H-PC'] = abs(df['high'] - df['close'].shift(1))
    df['L-PC'] = abs(df['low'] - df['close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1, skipna=False)
    df['ATR'] = df['TR'].rolling(n).mean()
    df_temp = df.drop(['H-L', 'H-PC', 'L-PC'], axis=1)
    return df_temp


# find local mimimum / local maximum
def is_LCC(DF, i):
    df = DF.copy()
    LCC = 0

    if df['close'][i] <= df['close'][i + 1] and df['close'][i] <= df['close'][i - 1] and df['close'][i + 1] > \
            df['close'][i - 1]:
        # найдено Дно
        LCC = i - 1
    return LCC


def is_HCC(DF, i):
    df = DF.copy()
    HCC = 0
    if df['close'][i] >= df['close'][i + 1] and df['close'][i] >= df['close'][i - 1] and df['close'][i + 1] < \
            df['close'][i - 1]:
        # найдена вершина
        HCC = i
    return HCC


def getMaxMinChannel(DF, n):
    maxx = 0
    minn = DF['low'].max()
    for i in range(1, n):
        if maxx < DF['high'][len(DF) - i]:
            maxx = DF['high'][len(DF) - i]
        if minn > DF['low'][len(DF) - i]:
            minn = DF['low'][len(DF) - i]
    return (maxx, minn)


def get_RSI(series, n=14):
    series = series.copy().diff()
    up, down = series.copy(), series.copy()
    up[up < 0] = 0
    down[down > 0] = 0

    roll_up1 = up.ewm(span=n, min_periods=0, adjust=False, ignore_na=False).mean()
    roll_down1 = down.abs().ewm(span=n, min_periods=0, adjust=False, ignore_na=False).mean()

    RSI = roll_up1 / roll_down1
    RSI1 = 100.0 - (100.0 / (1.0 + RSI))
    return (RSI1)


# generate data frame with all needed data
def PrepareDF(DF):
    ohlc = DF.iloc[:, [0, 1, 2, 3, 4, 5]]
    ohlc.columns = ["date", "open", "high", "low", "close", "volume"]
    ohlc = ohlc.set_index('date')
    df = ind_ATR(ohlc, 14).reset_index()
    df['slope'] = ind_slope(df['close'], 5)
    df['channel_max'] = df['high'].rolling(10).max()
    df['channel_min'] = df['low'].rolling(10).min()
    df['position_in_channel'] = (df['close'] - df['channel_min']) / (df['channel_max'] - df['channel_min'])
    df['RSI'] = get_RSI(df['close'], 14)
    df = df.set_index('date')
    df = df.reset_index()
    return (df)


def check_if_signal(symbol):
    ohlc = get_futures_klines(symbol, 100)
    prepared_df = PrepareDF(ohlc)
    i = 98  # 99 is current kandel which is not closed, 98 is last closed candel, we need 97 to check if it is bottom or top
    print(prepared_df['close'][i])
    print(prepared_df['RSI'][i])

    if is_LCC(prepared_df, i - 1) > 0:
        # found bottom - OPEN LONG
        if prepared_df['position_in_channel'][i - 1] < 0.5:
            # close to top of channel
            if prepared_df['slope'][i - 1] < -20:
                # found a good enter point for LONG
                print(prepared_df['close'][i], 'rsi', prepared_df['RSI'][i])
                signal = 'long'
                return signal

    if is_HCC(prepared_df, i - 1) > 0:
        # found top - OPEN SHORT
        if prepared_df['position_in_channel'][i - 1] > 0.5:
            # close to top of channel
            if prepared_df['slope'][i - 1] > 20:
                print(prepared_df['close'][i], 'rsi', prepared_df['RSI'][i])
                # found a good enter point for SHORT
                signal = 'short'
                return signal

def main(counterr):
    # try:
        signal = check_if_signal(symbol)
        if signal == 'long':
            print('----------------------------------\nlong\n----------------------------------')
        elif signal == 'short':
            print('----------------------------------\nshort\n----------------------------------')
    # except Exception as e:
    #     print(e)


starttime = time.time()
timeout = time.time() + 60 * 60 * 24  # 60 seconds times 60 meaning the script will run for 24 hr
counterr = 1

while time.time() <= timeout:
    try:
        print("script continue running at " + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        main(counterr)
        counterr = counterr + 1
        if counterr > 5:
            counterr = 1
        time.sleep(10 - ((time.time() - starttime) % 10.0))  # 1 minute interval between each new execution
    except KeyboardInterrupt:
        print('\n\KeyboardInterrupt. Stopping.')
        exit()


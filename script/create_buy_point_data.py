"""Create wave training data.
Automatically annotate data by identifying waves. The begining, 
middle and ending are obtained. Windows around these points are
collected as training data. They are organized in years.
"""
import sys, glob, os
path = os.getcwd()
sys.path.insert(0, ".")
from datetime import datetime
from vnpy.trader.database import database_manager
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from lib import utils
from lib.alg import get_waves


def normalize(x):
    #return x / (1e-9 + x.mean(2, keepdims=True))
    return x / (1e-9 + x[:, :, 0:1])

os.chdir(path)

WIN_SIZE = 10 # two weeks
SHIFT_PORTION = 5
FEATURE_KEYS = ['open_price', 'close_price', 'high_price', 'low_price', 'volume']

binfos = utils.fast_index().values
binfos = [b for b in binfos if b[3] == 'd'] # day line only
data_keys = ["buy", "sell", "hold", "empty"]
dic = {k: {} for k in data_keys}
buy_count = sell_count = hold_count = 0
for idx, binfo in enumerate(tqdm(binfos)):
    _, symbol, exchange, interval, _ = binfo
    vt_symbol = f"{symbol}.{exchange}"
    for key in data_keys:
        dic[key][vt_symbol] = {}

    start = datetime.strptime(f"2000-01-01", "%Y-%m-%d")
    end = datetime.strptime(f"2021-01-01", "%Y-%m-%d")

    bars = database_manager.load_bar_data_s(
        symbol=symbol, exchange=exchange, interval="d",
        start=start, end=end)
    if len(bars) < 100:
        continue

    df = utils.bars_to_df(bars)
    N = df.shape[0]

    # get waves
    waves = get_waves(df['close_price'], T1=0.30, T2=0.10)
    inter_waves = [[waves[i][2] + 1, waves[i+1][0] - 1] for i in range(len(waves) - 1)]

    points = {"buy": {}, "hold": {}, "sell": {}, "empty": {}}
    #times = {"buy": [], "hold": [], "sell": []}
    for year in range(2000, 2022):
        for key in points.keys():
            points[key][str(year)] = []

    for x1, y1, x2, y2, t in waves:
        # shift a portion around the starting point
        S = (x2 - x1) // SHIFT_PORTION
        win_st = max(x1, WIN_SIZE)
        win_ed = max(min(x1 + S, N), WIN_SIZE)
        for i in range(win_st + 1, win_ed + 1):
            d = np.array([df[key][i - WIN_SIZE : i] for key in FEATURE_KEYS])
            year = str(df.index[i - WIN_SIZE].year)
            points["buy"][year].append(d)
            #times["buy"].append(df.index[i - WIN_SIZE : i])
        
        win_st = max(x2 - S, WIN_SIZE)
        win_ed = min(x2, N)
        for i in range(win_st + 1, win_ed + 1):
            d = np.array([df[key][i - WIN_SIZE : i] for key in FEATURE_KEYS])
            year = str(df.index[i - WIN_SIZE].year)
            points["sell"][year].append(d)
            #times["sell"].append(df.index[i - WIN_SIZE : i])
        
        win_st = max(x1 + S, WIN_SIZE)
        win_ed = min(x2 - S, N)
        for i in range(win_st + 1, win_ed + 1):
            d = np.array([df[key][i - WIN_SIZE : i] for key in FEATURE_KEYS])
            year = str(df.index[i - WIN_SIZE].year)
            points["hold"][year].append(d)
            #times["hold"].append(df.index[i - WIN_SIZE : i])

    for st, ed in inter_waves:
        for i in range(st + 1, ed + 1):
            d = np.array([df[key][i - WIN_SIZE : i] for key in FEATURE_KEYS])
            year = str(df.index[i - WIN_SIZE].year)
            points["empty"][year].append(d)
            
    for key in points.keys():
        for year in points[key]:
            if len(points[key][year]) == 0:
                continue
            x = normalize(np.array(points[key][year]))
            if np.abs(x).max() < 100: # filter error data
                dic[key][vt_symbol][year] = x

    if (idx + 1) % 500 == 0:
        I = (idx + 1) // 500
        np.save(f"data/buy_point/share_{I}.npy", dic)
        del dic
        dic = {k: {} for k in data_keys}
np.save(f"data/buy_point/share_{I + 1}.npy", dic)
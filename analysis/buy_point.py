import sys, glob, os
path = os.getcwd()
sys.path.insert(0, ".")
from datetime import datetime
from vnpy.trader.database import database_manager
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import utils
from alg import get_waves

def normalize(x):
    return x / x[:, :, 0:1]
    

os.chdir(path)

WIN_SIZE = 10 # two weeks
SHIFT_PORTION = 3
FEATURE_KEYS = ['open_price', 'close_price', 'high_price', 'low_price', 'volume']

print("=> Loading all bar data")
binfos = database_manager.get_bar_data_statistics()

print("=> Forging data")
data_keys = ["buy", "sell", "hold"]
dic = {k: {} for k in data_keys}

for binfo in tqdm(binfos):
    symbol, exchange = binfo['symbol'], binfo['exchange']
    vt_symbol = f"{symbol}.{exchange}"
    for key in data_keys:
        dic[key][vt_symbol] = {}

    for year in range(2000, 2021):
        start = datetime.strptime(f"{year}-01-01", "%Y-%m-%d")
        end = datetime.strptime(f"{year}-01-01", "%Y-%m-%d")

        bars = database_manager.load_bar_data_s(
            symbol=symbol, exchange=exchange, interval="d",
            start=start, end=end)
        if len(bars) <= 100:
            continue
        df = utils.bars_to_df(bars)
        N = df.shape[0]
        # get waves
        waves = get_waves(df['close_price'], T1=0.20, T2=0.10)
        inter_waves = [[waves[i][2] + 1, waves[i+1][0] - 1] for i in range(len(waves) - 1)]
        # get labelings
        buy_points = []
        hold_points = []
        sell_points = []
        for x1, y1, x2, y2, t in waves:
            # shift 1/3 around the starting point
            win_st = max(x1 - (x2 - x1) / SHIFT_PORTION, WIN_SIZE)
            win_ed = max(min(int(x1 + (x2 - x1) / SHIFT_PORTION), N), WIN_SIZE)
            for i in range(win_st + 1, win_ed + 1):
                d = np.array([df[key][i - WIN_SIZE : i] for key in FEATURE_KEYS])
                buy_points.append(d)
            
            win_st = max(int(x1 + (SHIFT_PORTION - 1) * (x2 - x1) / SHIFT_PORTION), WIN_SIZE)
            win_ed = min(int(x2 + (x2 - x1) / SHIFT_PORTION), N)
            for i in range(win_st + 1, win_ed + 1):
                d = np.array([df[key][i - WIN_SIZE : i] for key in FEATURE_KEYS])
                sell_points.append(d)
            
            win_st = max(int(x1 + (x2 - x1) / SHIFT_PORTION), WIN_SIZE)
            win_ed = min(int(x1 + (SHIFT_PORTION - 1) * (x2 - x1) / SHIFT_PORTION), N)
            for i in range(win_st + 1, win_ed + 1):
                d = np.array([df[key][i - WIN_SIZE : i] for key in FEATURE_KEYS])
                hold_points.append(d)

        for st, ed in inter_waves:
            for i in range(st + 1, ed + 1):
                d = np.array([df[key][i - WIN_SIZE : i] for key in FEATURE_KEYS])
                hold_points.append(d)

        dic["buy"][vt_symbol][year] = normalize(np.array(buy_points))
        dic["sell"][vt_symbol][year] = normalize(np.array(sell_points))
        dic["hold"][vt_symbol][year] = normalize(np.array(hold_points))
    
np.save("buy_point.npy", dic)
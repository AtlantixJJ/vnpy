import sys, glob, os
path = os.getcwd()
sys.path.insert(0, ".")
from datetime import datetime
from vnpy.trader.database import database_manager
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import utils
from alg import get_waves


os.chdir(path)

WIN_SIZE = 10 # two weeks
FEATURE_KEYS = ['open_price', 'close_price', 'high_price', 'low_price', 'volume']

#binfos = database_manager.get_bar_statistics()
start = datetime.strptime("2000-01-01", "%Y-%m-%d")
end = datetime.strptime("2001-01-01", "%Y-%m-%d")

symbol = "000001"
exchange = "SZSE"

bars = database_manager.load_bar_data_s(
    symbol=symbol, exchange=exchange, interval="d",
    start=start, end=end)
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
    win_st = max(x1 - (x2 - x1) / 3, WIN_SIZE)
    win_ed = min(int(x1 + (x2 - x1) / 3), N)
    for i in range(win_st, win_ed):
        d = np.array([df[key][i - WIN_SIZE : i] for key in FEATURE_KEYS])
        buy_points.append(d)
    
    win_st = max(int(x1 + 2 * (x2 - x1) / 3), WIN_SIZE)
    win_ed = min(int(x2 + (x2 - x1) / 3), N)
    for i in range(win_st, win_ed):
        d = np.array([df[key][i - WIN_SIZE : i] for key in FEATURE_KEYS])
        sell_points.append(d)
    
    win_st = max(int(x1 + (x2 - x1) / 3), WIN_SIZE)
    win_ed = min(int(x1 + 2 * (x2 - x1) / 3), N)
    for i in range(win_st, win_ed):
        d = np.array([df[key][i - WIN_SIZE : i] for key in FEATURE_KEYS])
        hold_points.append(d)

for st, ed in inter_waves:
    for i in range(st, ed):
        d = np.array([df[key][i - WIN_SIZE : i] for key in FEATURE_KEYS])
        hold_points.append(d)

def normalize(x):
    return x / x[:, :, 0:1]

buy_points = normalize(np.array(buy_points))
sell_points = normalize(np.array(sell_points))
hold_points = normalize(np.array(hold_points))

print(buy_points.shape, sell_points.shape, hold_points.shape)
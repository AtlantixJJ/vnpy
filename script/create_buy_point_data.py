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
SHIFT_PORTION = 3
FEATURE_KEYS = ['open_price', 'close_price', 'high_price', 'low_price', 'volume']

binfos = utils.fast_index().values
binfos = [b for b in binfos if b[3] == 'd'] # day line only
data_keys = ["buy", "sell", "hold", "empty"]
key2color = {"buy": "red", "sell": "green", "hold": "orange", "empty": "blue"}
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
    prices = df['close_price'].values
    waves = get_waves(prices, T1=0.30, T2=0.10)
    # no interval between waves
    #inter_waves = []
    #for i in range(len(waves) - 1):
    #    bg, ed = waves[i][2] + 1, waves[i+1][0] - 1
    #    if ed - bg > 2:
    #        continue
    #    inter_waves.append((bg, ed))

    points = {"buy": {}, "hold": {}, "sell": {}, "empty": {}}
    #times = {"buy": [], "hold": [], "sell": []}
    for year in range(2000, 2022):
        for key in points.keys():
            points[key][str(year)] = []

    plot_flag = idx < 5
    plot_waves = 10
    if plot_flag:
        st = 0
        ed = waves[plot_waves - 1][2]
        x = np.arange(st, ed)
        y = df['close_price'].values[st:ed]
        plt.plot(x, y)
        lx = np.arange(st, ed)

    for wave_id, (x1, y1, x2, y2, t) in enumerate(waves):
        if t == -1: # decrease wave
            offset = 0.9
            start_key = "sell"
            middle_key = "empty"
            end_key = "empty"
        elif t == 0: # null wave
            offset = 1.0
            start_key = "empty"
            middle_key = "empty"
            end_key = "empty"
        elif t == 1: # increase wave
            offset = 1.1
            start_key = "buy"
            middle_key = "hold"
            end_key = "hold"
        #if plot_flag and wave_id < plot_waves:
        #    print(f"=> {wave_id} ({t}) : [{x1}, {x2}]")
        if plot_flag and t != 0:
            ly = (y2 - y1) * offset / (x2 - x1) * (lx - x1) + y1 * offset
        if plot_flag and t == 0:
            ly = np.zeros_like(lx) + y1 * offset

        # starting segment
        S = (x2 - x1) // SHIFT_PORTION
        win_st = max(x1, WIN_SIZE)
        win_ed = max(x1 + S, WIN_SIZE)
        if plot_flag and wave_id < plot_waves and win_ed > win_st:
            plt.plot(lx[win_st:win_ed], ly[win_st:win_ed],
                        color=key2color[start_key], linestyle=':')
        for i in range(win_st + 1, win_ed + 1):
            d = np.array([df[key][i - WIN_SIZE : i] for key in FEATURE_KEYS])
            year = str(df.index[i - WIN_SIZE].year)
            points[start_key][year].append(d)

        # middle segment
        win_st = max(x1 + S, WIN_SIZE)
        win_ed = min(x2 - S, N)
        if plot_flag and wave_id < plot_waves and win_ed > win_st:
            plt.plot(lx[win_st:win_ed], ly[win_st:win_ed],
                        color=key2color[middle_key], linestyle=':')
        for i in range(win_st + 1, win_ed + 1):
            d = np.array([df[key][i - WIN_SIZE : i] for key in FEATURE_KEYS])
            year = str(df.index[i - WIN_SIZE].year)
            points[middle_key][year].append(d)

        # end segment
        win_st = max(x2 - S, WIN_SIZE)
        win_ed = min(x2, N)
        if plot_flag and wave_id < plot_waves and win_ed > win_st:
            plt.plot(lx[win_st:win_ed], ly[win_st:win_ed],
                        color=key2color[end_key], linestyle=':')
        for i in range(win_st + 1, win_ed + 1):
            d = np.array([df[key][i - WIN_SIZE : i] for key in FEATURE_KEYS])
            year = str(df.index[i - WIN_SIZE].year)
            points[end_key][year].append(d)
            
    for key in points.keys():
        for year in points[key]:
            if len(points[key][year]) == 0:
                continue
            x = normalize(np.array(points[key][year]))
            if np.abs(x).max() < 100: # filter error data
                dic[key][vt_symbol][year] = x

    if plot_flag:
        plt.savefig(f"results/buy_point_viz_{idx}.png")
        plt.close()

    if (idx + 1) % 500 == 0:
        I = (idx + 1) // 500
        np.save(f"data/buy_point/share_{I}.npy", dic)
        del dic
        dic = {k: {} for k in data_keys}
np.save(f"data/buy_point/share_{I + 1}.npy", dic)
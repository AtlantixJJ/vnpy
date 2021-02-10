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
from lib.alg import label_waves


def normalize(x):
    #return x / (1e-9 + x.mean(2, keepdims=True))
    return x[:, :, 1:] / (1e-9 + x[:, :, :-1]) - 1


os.chdir(path)

WIN_SIZE = 20 # four weeks
PAD = 0 # the minimum distance between two different labels
NUM_SEGMENTS = 3
FEATURE_KEYS = ['open_price', 'close_price', 'high_price', 'low_price', 'volume']

binfos = utils.fast_index().values
binfos = [b for b in binfos if b[3] == 'd'] # day line only
data_keys = ["buy", "sell", "hold", "empty"]
key2color = {
    "buy": "red",
    "sell": "green",
    "hold": "orange",
    "empty": "blue"}
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
    waves, infos = label_waves(prices, T1=0.30, T2=0.20)

    points = {k: {} for k in data_keys}
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
        fig = plt.figure(figsize=(18, 12))
        ax1 = plt.subplot(2, 1, 1)
        ax1.plot(x, y)
        ax2 = plt.subplot(2, 1, 2)
        ax2.plot(infos[st:ed, 0])
        ax2.plot(infos[st:ed, 3])
        lx = np.arange(st, ed)

    for wave_id, (x1, y1, x2, y2, t) in enumerate(waves):
        if t == -1: # decrease wave
            offset = 0.8
            start_key, middle_key, end_key = "sell", "hold", "hold"
        elif t == 0: # null wave
            offset = 1.0
            start_key, middle_key, end_key = "empty", "empty", "empty"
        elif t == 1: # increase wave
            offset = 1.2
            start_key, middle_key, end_key = "buy", "hold", "hold"
        # segment length
        S = (x2 - x1) // NUM_SEGMENTS
        if plot_flag and t != 0:
            ly = (y2 - y1) * offset / (x2 - x1) * (lx - x1) + y1 * offset
        if plot_flag and t == 0:
            ly = np.zeros_like(lx) + y1 * offset

        def _work(ckey, win_st, win_ed):
            if win_st >= win_ed:
                return None
            for i in range(win_st, win_ed):
                d = np.array([df[key][i - WIN_SIZE : i] \
                        for key in FEATURE_KEYS])
                year = str(df.index[i - WIN_SIZE].year)
                points[ckey][year].append(d)

        if plot_flag and wave_id < plot_waves:
            ax1.plot(lx[x1:x2], ly[x1:x2],
                color=key2color[start_key], linestyle='-')

        _work(start_key, max(x1 + PAD + 1, WIN_SIZE), x1 + S + 1)
        _work(middle_key, max(x1 + S + PAD + 1, WIN_SIZE), x2 - S + 1)
        _work(end_key, max(x2 - S + PAD + 1, WIN_SIZE), x2 + 1)
            
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

    if (idx + 1) % 100 == 0:
        I = (idx + 1) // 100
        np.save(f"data/buy_point/share_{I:02d}.npy", dic)
        del dic
        dic = {k: {} for k in data_keys}
np.save(f"data/buy_point/share_{I + 1:02d}.npy", dic)
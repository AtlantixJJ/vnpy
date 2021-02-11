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
import pytz

from lib import utils
from lib.alg import label_waves


def normalize(x):
    #return x / (1e-9 + x.mean(2, keepdims=True))
    return x[:, :, 1:] / (1e-9 + x[:, :, :-1]) - 1


os.chdir(path)

T1 = 0.30
T2 = 0.15
WIN_SIZE = 20 # four weeks
PAD = 0 # the minimum distance between two different labels
NUM_SEGMENTS = 3
FEATURE_KEYS = ['open_price', 'close_price', 'high_price', 'low_price', 'volume']

binfos = utils.fast_index().values
binfos = [b for b in binfos if b[3] == 'd'] # day line only
data_keys = ["buy", "sell", "hold", "unknown"]
key2color = {
    "buy": "orange",
    "sell": "green",
    "hold": "blue",
    "unknown" : "cyan"}
dic = {year : {} for year in range(2000, 2021)}
year_data = np.load("data/year_data.npz", allow_pickle=True)['arr_0'][()]
for idx, binfo in enumerate(tqdm(binfos)):
    _, symbol, exchange, interval, _ = binfo
    vt_symbol = f"{symbol}.{exchange}"

    start = datetime.strptime(f"2000-01-01", "%Y-%m-%d")
    end = datetime.strptime(f"2021-01-01", "%Y-%m-%d")
    # start is included, end is not included
    bars = database_manager.load_bar_data_s(
        symbol=symbol, exchange=exchange, interval="d",
        start=start, end=end)
    if len(bars) < 100:
        continue

    df = utils.bars_to_df(bars)
    N = df.shape[0]

    # get waves
    prices = df['close_price'].values
    waves, infos = label_waves(prices, T1=T1, T2=T2)

    # store data according to years
    for year in range(int(df.index[0].year), 2021):
        start = datetime.strptime(f"{year}-01-01", "%Y-%m-%d")
        end = datetime.strptime(f"{year+1}-01-01", "%Y-%m-%d")
        st = df.index.searchsorted(start)
        ed = df.index.searchsorted(end)
        if ed - st == 0: continue
        #if vt_symbol not in year_data[year]:
        #    print(f"!> {vt_symbol} {year} data missing, but label is {ed - st}")
        shape = year_data[year][vt_symbol].shape
        dic[year][vt_symbol] = infos[st:ed]
        if shape[0] != ed - st:
            print(f"!> {vt_symbol} {year} shape mismatch: prices {shape}, info {ed - st}")
    
    if idx >= 5: # visualize the dataset for the first 5 instances
        continue

    # get labels
    P, R = infos[:, 0], infos[:, 1]
    labels = np.zeros((prices.shape[0],), dtype="uint8")
    # buy point: profits larger than threshold and 
    # retracts significantly lower than profits 
    buy_labels = (P > T2) & (-R < P / 2)
    # sell point: retracts is nonnegligiable and is larger than profits
    sell_labels = (-R > 0.05) & (-R > P / 2)
    # hold point: other cases
    hold_labels = (~buy_labels) & (~sell_labels)
    labels[buy_labels] = 0
    labels[sell_labels] = 1
    labels[hold_labels] = 2

    # plot examples of the dataset
    plot_flag, plot_waves = idx < 5, 10
    st, ed = 0, waves[plot_waves - 1][2]
    x = np.arange(st, ed)
    y = df['close_price'].values[st:ed]
    fig = plt.figure(figsize=(18, 12))
    ax1 = plt.subplot(2, 1, 1)
    colors = [key2color[data_keys[l]] for l in labels]
    utils.plot_multicolor_line(ax1, x, y, colors)
    ax1.set_title("Close Price & Wave")
    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(infos[st:ed, 0] * 100)
    ax2.plot(infos[st:ed, 1] * 100)
    ax2.set_title("Wave Profit & Retract (%)")
    lx = np.arange(st, ed)

    for wave_id, (x1, y1, x2, y2, t) in enumerate(waves[:plot_waves]):
        if t == -1: # decrease wave
            offset = 0.9
            ckey = "sell"
        elif t == 0: # null wave
            offset = 1.0
            ckey = "hold"
        elif t == 1: # increase wave
            offset = 1.1
            ckey = "buy"
        # segment length
        S = (x2 - x1) // NUM_SEGMENTS
        if t != 0:
            ly = (y2 - y1) * offset / (x2 - x1) * (lx - x1) + y1 * offset
        else:
            ly = np.zeros_like(lx) + y1 * offset

        ax1.plot(lx[x1:x2], ly[x1:x2],
            color=key2color[ckey], linestyle='-')
    plt.savefig(f"results/wave_pr_viz_{idx}.png")
    plt.close()
np.savez(f"data/wave_pr", dic)
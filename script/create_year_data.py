"""Create every's year's stock price data.
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


fpath = "data/year_data"
FEATURE_KEYS = ['open_price', 'close_price', 'high_price', 'low_price', 'volume']
binfos = utils.fast_index().values
binfos = [b for b in binfos if b[3] == 'd']
dic = {}
for year in range(1990, 2022):
    print(f"=> Processing year {year}")
    start = datetime.strptime(f"{year}-01-01", "%Y-%m-%d")
    end = datetime.strptime(f"{year+1}-01-01", "%Y-%m-%d")
    dic[year] = {}
    for idx, binfo in enumerate(tqdm(binfos)):
        _, symbol, exchange, interval, _ = binfo
        vt_symbol = f"{symbol}.{exchange}"
        bars = database_manager.load_bar_data_s(
            symbol=symbol, exchange=exchange, interval="d",
            start=start, end=end)
        if len(bars) == 0:
            continue
        df = utils.bars_to_df(bars)
        dic[year][vt_symbol] = df[FEATURE_KEYS].values
np.savez(fpath, dic)
import sys, glob, os
path = os.getcwd()
sys.path.insert(0, ".")
from datetime import datetime
from vnpy.trader.database import database_manager
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import utils

os.chdir(path)
bars = database_manager.load_bar_data_s(
    symbol="000001", exchange="SZSE", interval="d",
    start=datetime.strptime("2000-01-01", "%Y-%m-%d"),
    end=datetime.strptime("2021-1-30", "%Y-%m-%d"))

WIN = 21 # in a month: 31 * 250 / 365
df = utils.bars_to_df(bars)
v = df['close_price']

def get_waves(v, T1=0.2, T2=0.05, verbose=False):
    waves = []
    last_wave_idx = 0
    wave_type = 0
    lmax, lmin = v[0], v[0]
    lmaxi, lmini = 0, 0

    for i in range(1, v.shape[0]):
        if lmax < v[i]:
            lmax = v[i]
            lmaxi = i
        if lmin > v[i]:
            lmin = v[i]
            lmini = i
        
        if wave_type == 0 and v[i] > lmin * (1 + T1):
            if verbose:
                print(f"=> Start inc {lmin}({lmini}) -> {v[i]}({i})")
            wave_type = 1
            last_wave_idx = lmini
            lmax = v[i]
            lmaxi = i

        elif wave_type == 0 and v[i] < lmax * (1 - T1):
            if verbose:
                print(f"=> Start dec {lmax}({lmaxi}) -> {v[i]}({i})")
            wave_type = -1
            last_wave_idx = lmaxi
            lmin = v[i]
            lmini = i

        elif wave_type == 1 and v[i] < lmax * (1 - T2):
            if verbose:
                print(f"=> Inc -> Dec, wave {v[last_wave_idx]}({last_wave_idx}) ->{lmax}({lmaxi})")
            waves.append([last_wave_idx, lmaxi - 1, wave_type])
            last_wave_idx = lmaxi
            wave_type = 0
            lmin = v[i]
            lmini = i

        elif wave_type == -1 and v[i] > lmin * (1 + T2):
            if verbose:
                print(f"=> Dec -> Inc, wave {v[last_wave_idx]}({last_wave_idx}) ->{v[lmini - 1]}({lmini - 1})")
            waves.append([last_wave_idx, lmini - 1, wave_type])
            last_wave_idx = lmini
            wave_type = 0
            lmax = v[i]
            lmaxi = i

    waves.append([last_wave_idx, v.shape[0] - 1, wave_type])
    return waves

waves = get_waves(v)
st, ed = 2000, 2500
fig = plt.figure(figsize=(10, 5))
x = np.arange(0, v.shape[0])
plt.scatter(
    x[st:ed],
    v[st:ed], s=10)
plt.plot(x[st:ed], v[st:ed])
for lidx, cidx, t in waves:
    if cidx < st or lidx > ed or cidx > ed: continue
    c = 'r' if t == 1 else 'g'
    plt.plot([lidx, cidx], [v[lidx], v[cidx]], c)
plt.savefig("res.png")
plt.close()
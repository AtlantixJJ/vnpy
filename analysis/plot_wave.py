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
bars = database_manager.load_bar_data_s(
    symbol="000001", exchange="SZSE", interval="d",
    start=datetime.strptime("2000-01-01", "%Y-%m-%d"),
    end=datetime.strptime("2021-1-30", "%Y-%m-%d"))

WIN = 21 # in a month: 31 * 250 / 365
df = utils.bars_to_df(bars)
v = df['close_price']

waves = get_waves(v, T1=0.15, T2=0.05, verbose=True)
#print(waves)
st, ed = 0, 1000
fig = plt.figure(figsize=(10, 5))
x = np.arange(0, v.shape[0])
plt.scatter(
    x[st:ed],
    v[st:ed], s=10)
plt.plot(x[st:ed], v[st:ed])
for x1, y1, x2, y2, t in waves:
    if x2 < st or x1 > ed or x2 > ed: continue
    if t == 0: continue
    c = {0: 'blue', 1: 'red', -1: 'green'}[t]
    plt.plot([x1, x2], [y1, y2], c)
plt.savefig("res.png")
plt.close()
import sys, glob, os
path = os.getcwd()
sys.path.insert(0, ".")
from datetime import datetime
from vnpy.trader.constant import Interval, Exchange
from vnpy.app.data_manager import ManagerEngine
from vnpy.trader.engine import MainEngine, EventEngine
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

import utils


event_engine = EventEngine()
main_engine = MainEngine(event_engine)
data_manager = ManagerEngine(main_engine, event_engine)
os.chdir(path)
res = data_manager.get_bar_data_available()
binfo = res[0]
query = {v : binfo[v] for v in ['symbol', 'exchange', 'interval', 'start', 'end']}
bars = data_manager.load_bar_data_s(**query)

WIN = 21 # in a month: 31 * 250 / 365
df = utils.bars_to_df(bars)
v = df['close_price']
win_max = v.rolling(WIN).max()[WIN:].values
win_min = v.rolling(WIN).min()[WIN:].values
cv = v[:-WIN].values
win_max_r = (win_max - cv) / cv
win_min_r = (win_min - cv) / cv

stable_profit_buy = win_min_r > 0.05
high_profit_buy = (win_max_r > 0.20) & (win_min_r > -0.05)
sig_buy = stable_profit_buy | high_profit_buy
sig_sell = win_min_r < -0.1
res_map = {(True, True) : 'green', (True, False): 'red', (False, True): 'green', (False, False) : 'blue'}
x = list(range(len(v)))
c = [res_map[(sb, ss)] for sb, ss in zip(sig_buy, sig_sell)]
fig, ax1 = plt.subplots()
st, ed = 0, 100
ax1.scatter(
    x[st:ed],
    cv[st:ed],
    c=c[st:ed], s=10)
ax1.plot(x[st:ed], win_max[st:ed], 'r')
ax1.plot(x[st:ed], win_min[st:ed], 'g')
ax1.plot(x[st:ed], cv[st:ed])
#ax2 = ax1.twinx()
#ax2.plot(x[st:ed], win_max_r[st:ed], 'r-.')
#ax2.plot(x[st:ed], win_min_r[st:ed], 'g-.')
plt.savefig("res.png")
plt.close()
print("Done")
exit(0)
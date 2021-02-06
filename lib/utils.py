import pandas as pd
import numpy as np

import matplotlib
import matplotlib.style as style
style.use('seaborn-poster') #sets the size of the charts
style.use('ggplot')
import matplotlib.pyplot as plt
import numpy as np
import talib as ta


def plot_color_bar(ax, bars, mark):
    blues = np.where(mark == 0)[0]
    ax.bar(blues, bars[blues], color='b')
    reds = np.where(mark == 1)[0]
    ax.bar(reds, bars[reds], color='r')
    yellows = np.where(mark == 2)[0]
    ax.bar(yellows, bars[yellows], color='y')
    greens = np.where(mark == 3)[0]
    ax.bar(greens, bars[greens], color='g')


def plot_color_line(ax, mark):
    reds = np.where(mark == 1)[0]
    blues = np.where(mark == 2)[0]
    for idx in blues:
        ax.axvline(x=idx, color='b')
    for idx in reds:
        ax.axvline(x=idx, color='r')


def plot_dict_line(ax, dic, days):
    """Plot a line chart for a single value item."""
    val = dic["value"]
    # multiple lines
    if len(val) == 2:
        if "twin" in dic and dic["twin"] == True:
            ax.plot(val[0][-days:], 'r')
            ax2 = ax.twinx()
            ax2.plot(val[1][-days:], 'b')
        else:
            for line in val:
                ax.plot(line[-days:])
    # single line
    else:
        ax.plot(val[-days:])
    # draw markings
    if "mark" in dic:
        plot_color_line(ax, dic["mark"][-days:])


def plot_dict(dic, fpath, days=365):
    """
    Args:
        dic : { "<subplot title>" : { "value": [arrays] or array,
                "chart": str,
                ["twin": False], # optional
                ["mark": array],
                ["ma" : []] # moving average
                }}
              value is either a list of numpy arrays, or a single array.
              chart is type.
    """
    N = len(dic.keys()) # number of big graphs
    fig = plt.figure(figsize=(20, N * 3))

    for i, (group_key, v) in enumerate(dic.items()):
        val = v["value"]
        ax = plt.subplot(N, 1, i + 1)
        ax.set_title(group_key)

        if v["chart"] == "bar":
            plot_color_bar(ax, val[-days:], v["mark"][-days:])
        elif v["chart"] == "line": # line only 
            plot_dict_line(ax, v, days)
        # bar and line mixed chart
        elif "bar" in v["chart"] and "line" in v["chart"]:
            # bar
            plot_color_bar(ax, val[0][-days:], v["mark"][0][-days:])
            # twin line
            plot_dict_line(ax, v, days)
        else:
            t = v["chart"]
            print(f"!> Chart type {t} not understood")

        # Add moving average
        if "ma" in v:
            mas = [ta.SMA(val[-days-p:], timeperiod=p) for p in v["ma"]]
            for ma in mas:
                ax.plot(ma[-days:])
    plt.savefig(fpath)
    plt.close()


def fast_index(fpath="data/index.csv"):
    """Return the fast index of the database"""
    import pandas
    return pandas.read_csv(fpath, dtype=str)

def bars_to_df(bars):
    keys = ['open_price' , 'high_price', 'low_price', 'close_price', 'volume']
    data = np.array([[getattr(b, k) for k in keys] for b in bars])
    return pd.DataFrame(
        data=data,
        index=[b.datetime for b in bars],
        columns=keys)
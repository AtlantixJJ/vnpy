import glob, torch
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.style as style
style.use('seaborn-poster') #sets the size of the charts
style.use('ggplot')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import talib as ta


#########################
##### visualization #####
#########################


def label2color(y, color_dict={0: 'b', 1: 'r', 2: 'g'}):
    return [color_dict[int(num)] for num in y]


def plot_multicolor_line(ax=None, x=None, y=None, colors=None):
    if x is None:
        x = np.arange(y.shape[0])
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], 1)
    lc = LineCollection(segments, colors=colors)
    ax.add_collection(lc)
    ax.set_xlim([x.min(), x.max()])
    ax.set_ylim([y.min(), y.max()])


def value_from_pdeltas(pdeltas):
    """Calculate true value from percentage deltas."""
    vals = np.ones_like(pdeltas)
    for i in range(1, pdeltas.shape[0]):
        vals[i] = (1 + pdeltas[i] / 100.) * vals[i - 1]
    return vals


def plot_wave_pr(feats, infos, y_true, y_pred):
    """Assume the batch dim is 1."""
    feats = feats.numpy()
    infos = infos.numpy()
    y_true = y_true.numpy()
    y_pred = y_pred.numpy()
    return plot_dict({ "Price" : {
            "value" : value_from_pdeltas(feats[0, :, 1]),
            "chart" : "mcline",
            "mark" : y_true[0]
        }, "Delta Percentages" : {
            "value" : feats[0, :, 1],
            "chart" : "mcline",
            "mark" : y_pred[0]
        }, "Volume" : {
            "value" : value_from_pdeltas(feats[0, :, 4]),
            "chart" : "bar",
            "mark" : (feats[0, :, 4] > 0).astype("uint8")
        }, "Wave Profit & Retract (%)" : {
            "value" : [infos[0, :, 0], infos[0, :, 1]],
            "chart" : "line"
        }})


def plot_color_bar(ax, bars, mark):
    blues = np.where(mark == 0)[0]
    ax.bar(blues, bars[blues], color='b')
    reds = np.where(mark == 1)[0]
    ax.bar(reds, bars[reds], color='r')
    yellows = np.where(mark == 2)[0]
    ax.bar(yellows, bars[yellows], color='y')
    greens = np.where(mark == 3)[0]
    ax.bar(greens, bars[greens], color='g')


def plot_color_vmark(ax, mark):
    reds = np.where(mark == 1)[0]
    blues = np.where(mark == 2)[0]
    for idx in blues:
        ax.axvline(x=idx, color='b')
    for idx in reds:
        ax.axvline(x=idx, color='r')


def plot_dict_line(ax, dic, last_N):
    """Plot a line chart for a single value item."""
    val = dic["value"]
    # multiple lines
    if len(val) == 2:
        if "twin" in dic and dic["twin"] == True:
            ax.plot(val[0][-last_N:], 'r')
            ax2 = ax.twinx()
            ax2.plot(val[1][-last_N:], 'b')
        else:
            for line in val:
                ax.plot(line[-last_N:])
    # single line
    else:
        ax.plot(val[-last_N:])
    # draw markings
    if "mark" in dic:
        plot_color_vmark(ax, dic["mark"][-days:])


def plot_dict(dic, last_N=500):
    """Plot figures using a dict format.
    Args:
        dic:{ "<subplot title>" : {
                "value": List[array] or array,
                "chart": str, # bar, line, bar-line, mcline
                ["twin": False], # optional
                ["mark": array],
                ["ma" : []] # moving average
              }
            }
              value is either a list of numpy arrays, or a single array.
              chart is type.
    """
    N = len(dic.keys()) # number of big graphs
    fig = plt.figure(figsize=(20, N * 3))

    for i, (group_key, item_dic) in enumerate(dic.items()):
        val = item_dic["value"]
        ax = plt.subplot(N, 1, i + 1)
        ax.set_title(group_key)

        if item_dic["chart"] == "bar":
            plot_color_bar(ax, val[-last_N:], item_dic["mark"][-last_N:])
        elif item_dic["chart"] == "line": # line only 
            plot_dict_line(ax, item_dic, last_N)
        elif item_dic["chart"] == "mcline":
            plot_multicolor_line(ax=ax, y=val[-last_N:],
                colors=label2color(item_dic["mark"]))
        # bar and line mixed chart
        elif "bar" in item_dic["chart"] and "line" in item_dic["chart"]:
            # bar
            plot_color_bar(ax, val[0][-last_N:], item_dic["mark"][0][-last_N:])
            # twin line
            plot_dict_line(ax, item_dic, last_N)
        else:
            t = item_dic["chart"]
            print(f"!> Chart type {t} not understood")

        # Add moving average
        if "ma" in item_dic:
            mas = [ta.SMA(val[-last_N-p:], timeperiod=p) for p in item_dic["ma"]]
            for ma in mas:
                ax.plot(ma[-last_N:])
    return fig


###########################
##### data processing #####
###########################


def parse_years(year_range):
    """Parse year_range into year list.
    Args:
        year_range: A string in the format aaaa-bbbb.
    Returns:
        A list of years from aaaa to bbbb, including both ends.
    """
    st, ed = year_range.split("-")
    st, ed = int(st), int(ed)
    return [year for year in range(st, ed + 1)]


def split(data, train_ratio, val_ratio, seed=None):
    """Split a data into train / val / test split."""
    N = data.shape[0]
    test_ratio = 1 - train_ratio - val_ratio
    rng = np.random.RandomState(seed if seed else 1)
    indice = np.arange(N)
    rng.shuffle(indice)
    st, ed = 0, int(N * train_ratio)
    train_indice = indice[:ed]
    st, ed = ed, int(N * (train_ratio + val_ratio))
    val_indice = indice[st:ed]
    test_indice = indice[ed:]
    return data[train_indice], data[val_indice], data[test_indice]


def tensor_from_dict2d(dic, keys1, keys2, delete=True):
    """Create a sequential Tensor from dic, given keys."""
    n_symbols = len(dic.keys())
    data = []
    count = 0
    for k1 in keys1:
        for k2 in keys2:
            if k2 in dic[k1]:
                N, FEAT_SIZE, WIN_SIZE = dic[k1][k2].shape
                count += N
    if count == 0:
        print("!> Dataset is empty")
        return None
    x = torch.FloatTensor(count, FEAT_SIZE, WIN_SIZE)
    count = 0
    for k1 in keys1:
        for k2 in keys2:
            if k2 in dic[k1]:
                N = dic[k1][k2].shape[0]
                x[count : count + N] = torch.from_numpy(dic[k1][k2])
                count += N
                del dic[k1][k2]
    return x


def balance_class(xs, ys, max_dev=1.1):
    """Balancing each category.
    Only re-sample the most imbalanced class.
    """
    lens = np.array([y.shape[0] for y in ys])
    ratios = lens / lens.sum().astype("float32")
    indice = np.argsort(lens)
    maxi2, maxi1 = indice[-2:]
    if ratios[maxi1] > max_dev * ratios[maxi2]:
        sample_rate = max_dev * lens[maxi2] / lens[maxi1]
        N_total = xs[maxi1].shape[0]
        length = int(sample_rate * N_total)
        new_ratio = length / (lens.sum() - N_total + length)
        print(f"=> Balance resample class {maxi1} from {ratios[maxi1]*100:.2f}% to {new_ratio*100:.2f}%")
        rng = np.random.RandomState(1)
        indice = rng.choice(np.arange(0, N_total),
            size=(length,), replace=False)
        xs[-1] = xs[-1][indice]
        ys[-1] = ys[-1][indice]
    return xs, ys


def load_year(res, data_dir="data/buy_point", years=[2000]):
    files = glob.glob(f"{data_dir}/share*")
    N = len(files)

    def assign(data_key, symbol, year):
        res[data_key][symbol][year] = dic[data_key][symbol][year]

    for i in range(1, N + 1):
        fpath = f"{data_dir}/share_{i}.npy"
        print(f"=> Loading {fpath}")
        dic = np.load(fpath, allow_pickle=True)[()]
        for data_key in dic.keys():
            if data_key not in res:
                res[data_key] = {}
            for symbol in dic[data_key]:
                if symbol not in res[data_key]:
                    res[data_key][symbol] = {}
                for year in years:
                    year = str(year)
                    if year not in dic[data_key][symbol]:
                        continue
                    assign(data_key, symbol, year)


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
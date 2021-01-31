import pandas as pd
import numpy as np

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
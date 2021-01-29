import sys, glob, os
path = os.getcwd()
sys.path.insert(0, ".")
from pytdx.reader import TdxDailyBarReader, TdxMinBarReader
from datetime import datetime
from vnpy.trader.constant import Interval, Exchange
from vnpy.app.data_manager import ManagerEngine
from vnpy.trader.engine import MainEngine, EventEngine
from tqdm import tqdm


event_engine = EventEngine()
main_engine = MainEngine(event_engine)
data_manager = ManagerEngine(main_engine, event_engine)
os.chdir(path)
DATA_DIR = "data"
daily_reader = TdxDailyBarReader()
min_reader = TdxMinBarReader()
bars = data_manager.get_bar_data_available()
symbols = set([b['symbol'] for b in bars if b['interval'] == '5min'])
#print(symbols)

for data_dir in ["sz5fz", "sh5fz"]:
    exchange = Exchange.SSE if data_dir == "sh5fz" else Exchange.SZSE
    files = glob.glob(os.path.join(DATA_DIR, data_dir, "*"))
    files.sort()
    for f in tqdm(files):
        name = f[-8:-2]
        if name[0] not in "036":
            continue

        try:
            df = min_reader.get_df(f)
        except:
            print(f"!> skip {f}")
            continue

        if df.shape[0] <= 100:
            print(f"=> Skip {name}. Its shape is {df.shape}")
            continue
        data_manager.import_data_from_dict(
            reader=df,
            symbol=name,
            exchange=exchange,
            interval=Interval.FIVEMIN,
            datetime_head='date',
            open_head='open',
            high_head='high',
            low_head='low',
            close_head='close',
            volume_head='volume',
            open_interest_head='',
            datetime_format='%Y-%m-%d')
        


symbols = set([b['symbol'] for b in bars])
for data_dir in ["szlday", "shlday"]:
    exchange = Exchange.SSE if data_dir == "shlday" else Exchange.SZSE
    files = glob.glob(f"{DATA_DIR}/{data_dir}/*")
    files.sort()
    for f in tqdm(files):
        name = f[-10:-4]
        if name in symbols or name[0] not in "036":
            continue

        try:
            df = daily_reader.get_df(f)
        except:
            print(f"!> skip {f}: reading failed")
            continue

        if df.shape[0] <= 100:
            print(f"=> Skip {name}: {df.shape}")
            continue

        data_manager.import_data_from_dict(
            reader=df,
            symbol=name,
            exchange=exchange,
            interval=Interval.DAILY,
            datetime_head='date',
            open_head='open',
            high_head='high',
            low_head='low',
            close_head='close',
            volume_head='volume',
            open_interest_head='',
            datetime_format='%Y-%m-%d')



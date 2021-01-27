import sys, glob, os
path = os.getcwd()
sys.path.insert(0, ".")
from pytdx.reader import TdxDailyBarReader, TdxFileNotFoundException
from datetime import datetime
from vnpy.trader.constant import Interval, Exchange
from vnpy.app.data_manager import ManagerEngine
from vnpy.trader.engine import MainEngine, EventEngine
from tqdm import tqdm

event_engine = EventEngine()
main_engine = MainEngine(event_engine)
data_manager = ManagerEngine(main_engine, event_engine)
bars = data_manager.get_bar_data_available()
symbols = set([b['symbol'] for b in bars])
print(symbols)
os.chdir(path)
DATA_DIR = "data"
interval = Interval.DAILY
reader = TdxDailyBarReader()
for data_dir in ["szlday", "shlday"]:
    exchange = Exchange.SSE if data_dir == "shlday" else Exchange.SZSE
    files = glob.glob(f"{DATA_DIR}/{data_dir}/*")
    files.sort()
    for f in tqdm(files):
        name = f[-10:-4]
        if name in symbols:
            print(f"=> Skip {name}")
            continue
        try:
            df = reader.get_df(f)
        except:
            print(f"!> skip {f}")
        data_manager.import_data_from_dict(
            reader=df,
            symbol=name,
            exchange=exchange,
            interval=interval,
            datetime_head='date',
            open_head='open',
            high_head='high',
            low_head='low',
            close_head='close',
            volume_head='volume',
            open_interest_head='',
            datetime_format='%Y-%m-%d')

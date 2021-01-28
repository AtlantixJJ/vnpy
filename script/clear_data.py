import sys, glob, os
path = os.getcwd()
sys.path.insert(0, ".")
from vnpy.trader.database import database_manager

bars = database_manager.get_bar_data_statistics()
for b in bars:
    if b['symbol'][0] not in "036":
        database_manager.delete_bar_data(b['symbol'], b['exchange'], b['interval'])
        print(f"=> Delete {b}")

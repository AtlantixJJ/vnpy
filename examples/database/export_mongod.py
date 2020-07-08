from datetime import datetime

from vnpy.trader.ui import create_qapp, QtCore
from vnpy.trader.database import database_manager
from vnpy.trader.constant import Exchange, Interval
from vnpy.chart import ChartWidget, VolumeItem, CandleItem
from vnpy.app.data_manager import ManagerEngine
from vnpy.trader.engine import MainEngine, EventEngine

if __name__ == "__main__":
  app = create_qapp()
  name = "600031.SH_三一重工"
  me = ManagerEngine(MainEngine, EventEngine)
  me.output_data_to_csv(
    f"/Users/jianjinxu/python/waditu/day_K_north/{name}.csv",
    name,
    Exchange.SSE,
    Interval.DAILY,
    '20140101',
    '20210101')
from datetime import datetime

from vnpy.trader.ui import create_qapp, QtCore
from vnpy.trader.database import database_manager
from vnpy.trader.constant import Exchange, Interval
from vnpy.chart import ChartWidget, VolumeItem, CandleItem
from vnpy.app.data_manager import ManagerEngine

if __name__ == "__main__":
  app = create_qapp()
  name = "600031.SH_三一重工"
  ManagerEngine.import_data_from_csv(
    "",
    f"/Users/jianjinxu/python/waditu/day_K_north/{name}.csv",
    name,
    Exchange.SSE,
    Interval.DAILY,
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "north_delta",
    "amount",
    "%Y%m%d")
  
  bars = database_manager.load_bar_data(
    name,
    Exchange.SSE,
    Interval.DAILY,
    '20140101',
    '20210101')

  widget = ChartWidget()
  widget.add_plot("candle", hide_x_axis=True)
  widget.add_plot("volume", maximum_height=200)
  widget.add_item(CandleItem, "candle", "candle")
  widget.add_item(VolumeItem, "volume", "volume")
  widget.add_cursor()

  widget.update_history(bars)
  widget.show()
  app.exec_()

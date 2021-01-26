from datetime import datetime
import sys
sys.path.insert(0, ".")
from vnpy.trader.ui import create_qapp, QtCore
from vnpy.trader.database import database_manager
from vnpy.trader.constant import Exchange, Interval
from vnpy.chart import ChartWidget, VolumeItem, CandleItem

if __name__ == "__main__":
    app = create_qapp()

    bars = database_manager.load_bar_data(
        "sh000003",
        Exchange.SSE,
        interval=Interval.DAILY,
        start=datetime(2020, 1, 1),
        end=datetime(2021, 1, 1))

    widget = ChartWidget()
    widget.add_plot("candle", hide_x_axis=True)
    widget.add_plot("volume", maximum_height=200)
    widget.add_item(CandleItem, "candle", "candle")
    widget.add_item(VolumeItem, "volume", "volume")
    widget.add_cursor()

    n = 1000
    history = bars[:n]
    new_data = bars[n:]

    widget.update_history(history)

    def update_bar():
        bar = new_data.pop(0)
        widget.update_bar(bar)

    timer = QtCore.QTimer()
    timer.timeout.connect(update_bar)
    timer.start(100)
    print("show")
    widget.show()
    app.exec_()

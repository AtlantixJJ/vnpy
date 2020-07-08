import tushare as ts
from vnpy.trader.constant import Interval, Exchange
from vnpy.trader.object import BarData
from mongoengine import DateTimeField, Document, IntField, FloatField, StringField, connect
from vnpy.trader.database.database_mongo import MongoManager


TOKEN = 'b1de187329ab1435c5940f191db32539c66bdad30db3eb06a512ba70'
ts.set_token(TOKEN)
pro = ts.pro_api()

KLINE_INFO = ["open", "high", "low", "close", "pre_close", "change", "vol","amount"]

class EquityDailyInfo(Document):
  def __init__(self):
    self.value_pair = {}
    self.meta = {
      "indexes": [
        {
          "fields": ("symbol", "datetime"),
          "unique": True,
        }
      ]}

  def key_update(self, df):
    for k in df.keys():
      if not hasattr(self, k):
        t = StringField
        if type(df[k][0]) is int:
          t = IntField
        elif type(df[k][0]) is float:
          t = FloatField
        setattr(self, k, t())
        self.value_pair[k] = getattr(self, k)


def download_day_K_history(symbol: str):
  df = ts.pro_bar(
    ts_code=symbol, # 012345.SH
    asset='E',
    adj='qfq')
  return df


def download_daily_basic(symbol: str):
  df = pro.daily_basic(ts_code=symbol)
  return df


def download_income_table(symbol: str):
  df = pro.income(ts_code=symbol)
  return df


def download_balance_sheet(symbol: str):
  df = pro.balancesheet(ts_code=symbol)
  return df


def download_cashflow(symbol: str):
  df = pro.cashflow(ts_code=symbol)
  return df


def download_finance_indicator(symbol: str):
  df = pro.fina_indicator(ts_code=symbol)
  return df


def daily_info_object(d):
  return EquityDailyInfo.objects(**d)

a = download_daily_basic("000001.SZ")
b = EquityDailyInfo()
b.key_update(a)

mm = MongoManager()
daily_info_object(a.as_matrix() )

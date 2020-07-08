import tushare as ts
from vnpy.trader.constant import Interval, Exchange
from vnpy.trader.object import BarData
from vnpy.trader.database.database_mongo import MongoManager
from datetime import datetime
import numpy as np
from vnpy.trader.tstype import *
from tqdm import tqdm
import os


TOKEN = 'b1de187329ab1435c5940f191db32539c66bdad30db3eb06a512ba70'
ts.set_token(TOKEN)
pro = ts.pro_api()
mm = MongoManager()


def update_mongod_trade_date(df, D):
  for value in df.values:
    dic = {k: value[i] for i, k in enumerate(df.keys())}

    ts_code = dic.pop('ts_code')
    trade_date = datetime.strptime(dic.pop('trade_date'), "%Y%m%d")
    obj = D.objects(
      ts_code=ts_code,
      trade_date=trade_date)
    updates = mm.to_update_param(dic)
    obj.update_one(upsert=True, **updates)


def update_mongod_end_date(df, D):
  for value in df.values:
    dic = {k: value[i] for i, k in enumerate(df.keys())}

    ts_code = dic.pop('ts_code')
    end_date = datetime.strptime(dic.pop('end_date'), "%Y%m%d")
    obj = D.objects(
      ts_code=ts_code,
      end_date=end_date)
    updates = mm.to_update_param(dic)
    obj.update_one(upsert=True, **updates)


def update_mongod(df, D):
  for value in df.values:
    dic = {k: value[i] for i, k in enumerate(df.keys())}

    ts_code = dic.pop('ts_code')
    obj = D.objects(ts_code=ts_code)
    updates = mm.to_update_param(dic)
    obj.update_one(upsert=True, **updates)

### Permanent information

def download_equity_basic():
  df = pro.stock_basic(
    list_status='L',
    fields='ts_code,name,area,industry,market,is_hs')
  update_mongod(df, EquityBasic)
  return df

### daily information

def download_day_K(symbol: str, start_date='', end_date=''):
  df = ts.pro_bar(
    ts_code=symbol, # 012345.SH
    asset='E',
    start_date=start_date,
    end_date=end_date,
    adj='qfq')
  update_mongod_trade_date(df, EquityDayKInfo)
  return df


def download_daily_basic(symbol: str, start_date='', end_date=''):
  df = pro.daily_basic(
    ts_code=symbol,
    start_date=start_date,
    end_date=end_date)
  update_mongod_trade_date(df, EquityDailyInfo)
  return df


def download_north(symbol: str, start_date='', end_date=''):
  df = pro.hk_hold(
    ts_code=symbol,
    start_date=start_date,
    end_date=end_date,
    fields=['trade_date', 'ts_code', 'vol', 'ratio'])
  update_mongod_trade_date(df, EquityNorthCapitalInfo)
  return df

### Quater information

def download_income_table(symbol: str):
  df = pro.income(ts_code=symbol)
  update_mongod_end_date(df, EquityIncomeInfo)
  return df


def download_balance_sheet(symbol: str):
  df = pro.balancesheet(ts_code=symbol)
  update_mongod_end_date(df, EquityBalanceSheetInfo)
  return df


def download_cashflow(symbol: str):
  df = pro.cashflow(ts_code=symbol)
  update_mongod_end_date(df, EquityCashFlowInfo)
  return df


def download_finance_indicator(symbol: str):
  df = pro.fina_indicator(ts_code=symbol)
  update_mongod_end_date(df, EquityFinanceIndicatorInfo)
  return df


def synchronize_historical_data():
  # equity basics
  download_equity_basic()
  objs = EquityBasic.objects()

  # arrange to get the maximum bandwidth
  mask = np.array([obj.is_hs == "S" for obj in objs])
  sidx = np.where(mask)[0]
  nidx = np.where(~mask)[0]
  idxs = []
  c1, c2 = 0, 0
  for i in range(len(mask)):
    if i % 5 == 0:
      idxs.append(sidx[c1])
      c1 += 1
    else:
      if c2 >= len(nidx):
        idxs.append(-1)
      else:
        idxs.append(nidx[c2])
        c2 += 1

  for idx in tqdm(idxs):
    if idx < 0:
      os.system("sleep 8")
      continue
    obj = objs[int(idx)]
    print(f"=> Downloading {obj.name}")
    if obj.is_hs == "S":
      # 沪港通、深港通，获取北向资金
      print("=> Calling HS")
      north_capital = download_north(obj.ts_code)
    download_day_K(obj.ts_code)
    download_daily_basic(obj.ts_code)
    download_balance_sheet(obj.ts_code)
    download_cashflow(obj.ts_code)
    download_finance_indicator(obj.ts_code)
    download_income_table(obj.ts_code)
    
    
    

if __name__ == "__main__":
  ts_code = "000002.SZ"
  #print_key_type(get_key_type(daily_basic_dic))
  #download_daily_basic(ts_code)
  #download_day_K_history(ts_code)

  #print("Stock: \n")
  #df = download_equity_basic()
  #print(df.keys())
  #print(df.values[0])
  #print_key_type(get_key_type(df))

  #print("North: \n")
  #df = download_north(ts_code)
  #print_key_type(get_key_type(df))

  #print("Income:\n")
  #df = download_income_table(ts_code)
  #print_key_type(get_key_type(df))

  #print("Balance:\n")
  #df = download_balance_sheet(ts_code)
  #print_key_type(get_key_type(df))

  #print("Cashflow:\n")
  #df = download_cashflow(ts_code)
  #print_key_type(get_key_type(df))

  #print("Fina:\n")
  #df = download_finance_indicator(ts_code)
  #print_key_type(get_key_type(df))


  """
  trade_date = datetime.strptime('20200703', "%Y%m%d")
  obj = EquityDailyInfo.objects(
      ts_code=ts_code,
      trade_date=trade_date)
  print(obj)
  obj = EquityDayKInfo.objects(
      ts_code=ts_code,
      trade_date=trade_date)
  print(obj)
  """

  synchronize_historical_data()
  
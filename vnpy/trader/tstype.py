from mongoengine import DateTimeField, Document, IntField, FloatField, StringField
import numpy as np
from datetime import datetime


def get_key_type(df):
  dic = {}
  for k in df.keys():
    t = "FloatField()"
    if type(df[k][0]) is np.int64:
      t = "IntField()"
    elif type(df[k][0]) is np.float64:
      t = "FloatField()"
    elif type(df[k][0]) is str:
      t = "StringField()"
    dic[k] = t
  return dic


def print_key_type(dic):
  for k, v in dic.items():
    print(f"{k} = {v}")


class EquityBasic(Document):
  meta = {
    "indexes": [
      {
        "fields": ("ts_code",),
        "unique": True,
      }
    ]}

  ts_code = StringField()
  name = StringField()
  area = StringField()
  industry = StringField()
  market = StringField()
  is_hs = StringField()

  def __str__(self):
    f = ""
    for k in dir(self):
      if k[0] != '_' and hasattr(self, k):
        v = getattr(self, k)
        if type(v) in [float, str, int, datetime]:
          f = f + f"{k} = {v}\n"
    return f

  def __repr__(self):
    return str(self)


class EquityNorthCapitalInfo(Document):
  meta = {
    "indexes": [
      {
        "fields": ("ts_code", "trade_date"),
        "unique": True,
      }
    ]}

  ts_code = StringField()
  trade_date = DateTimeField()

  vol = IntField()
  ratio = FloatField()

  def __str__(self):
    f = ""
    for k in dir(self):
      if k[0] != '_' and hasattr(self, k):
        v = getattr(self, k)
        if type(v) in [float, str, int, datetime]:
          f = f + f"{k} = {v}\n"
    return f

  def __repr__(self):
    return str(self)


class EquityDailyInfo(Document):
  meta = {
    "indexes": [
      {
        "fields": ("ts_code", "trade_date"),
        "unique": True,
      }
    ]}

  ts_code = StringField()
  trade_date = DateTimeField()

  close = FloatField()
  turnover_rate = FloatField()
  turnover_rate_f = FloatField()
  volume_ratio = FloatField()
  pe = FloatField()
  pe_ttm = FloatField()
  pb = FloatField()
  ps = FloatField()
  ps_ttm = FloatField()
  dv_ratio = FloatField()
  dv_ttm = FloatField()
  total_share = FloatField()
  float_share = FloatField()
  free_share = FloatField()
  total_mv = FloatField()
  circ_mv = FloatField()

  def __str__(self):
    f = ""
    for k in dir(self):
      if k[0] != '_' and hasattr(self, k):
        v = getattr(self, k)
        if type(v) in [float, str, int, datetime]:
          f = f + f"{k} = {v}\n"
    return f

  def __repr__(self):
    return str(self)


class EquityDayKInfo(Document):
  meta = {
    "indexes": [
      {
        "fields": ("ts_code", "trade_date"),
        "unique": True,
      }
    ]}

  ts_code = StringField()
  trade_date = StringField()
  open = FloatField()
  high = FloatField()
  low = FloatField()
  close = FloatField()
  pre_close = FloatField()
  change = FloatField()
  pct_chg = FloatField()
  vol = FloatField()
  amount = FloatField()

  def __str__(self):
    f = ""
    for k in dir(self):
      if k[0] != '_' and hasattr(self, k):
        v = getattr(self, k)
        if type(v) in [float, str, int, datetime]:
          f = f + f"{k} = {v}\n"
    return f

  def __repr__(self):
    return str(self)


class EquityIncomeInfo(Document):
  meta = {
    "indexes": [
      {
        "fields": ("ts_code", "end_date"),
        "unique": True,
      }
    ]}

  ts_code = StringField()
  ann_date = DateTimeField()
  f_ann_date = DateTimeField()
  end_date = DateTimeField()
  report_type = StringField()
  comp_type = StringField()
  basic_eps = FloatField()
  diluted_eps = FloatField()
  total_revenue = FloatField()
  revenue = FloatField()
  int_income = FloatField()
  prem_earned = FloatField()
  comm_income = FloatField()
  n_commis_income = FloatField()
  n_oth_income = FloatField()
  n_oth_b_income = FloatField()
  prem_income = FloatField()
  out_prem = FloatField()
  une_prem_reser = FloatField()
  reins_income = FloatField()
  n_sec_tb_income = FloatField()
  n_sec_uw_income = FloatField()
  n_asset_mg_income = FloatField()
  oth_b_income = FloatField()
  fv_value_chg_gain = FloatField()
  invest_income = FloatField()
  ass_invest_income = FloatField()
  forex_gain = FloatField()
  total_cogs = FloatField()
  oper_cost = FloatField()
  int_exp = FloatField()
  comm_exp = FloatField()
  biz_tax_surchg = FloatField()
  sell_exp = FloatField()
  admin_exp = FloatField()
  fin_exp = FloatField()
  assets_impair_loss = FloatField()
  prem_refund = FloatField()
  compens_payout = FloatField()
  reser_insur_liab = FloatField()
  div_payt = FloatField()
  reins_exp = FloatField()
  oper_exp = FloatField()
  compens_payout_refu = FloatField()
  insur_reser_refu = FloatField()
  reins_cost_refund = FloatField()
  other_bus_cost = FloatField()
  operate_profit = FloatField()
  non_oper_income = FloatField()
  non_oper_exp = FloatField()
  nca_disploss = FloatField()
  total_profit = FloatField()
  income_tax = FloatField()
  n_income = FloatField()
  n_income_attr_p = FloatField()
  minority_gain = FloatField()
  oth_compr_income = FloatField()
  t_compr_income = FloatField()
  compr_inc_attr_p = FloatField()
  compr_inc_attr_m_s = FloatField()
  ebit = FloatField()
  ebitda = FloatField()
  insurance_exp = FloatField()
  undist_profit = FloatField()
  distable_profit = FloatField()

  def __str__(self):
    f = ""
    for k in dir(self):
      if k[0] != '_' and hasattr(self, k):
        v = getattr(self, k)
        if type(v) in [float, str, int, datetime]:
          f = f + f"{k} = {v}\n"
    return f

  def __repr__(self):
    return str(self)


class EquityFinanceIndicatorInfo(Document):
  meta = {
    "indexes": [
      {
        "fields": ("ts_code", "end_date"),
        "unique": True,
      }
    ]}

  ts_code = StringField()
  ann_date = DateTimeField()
  end_date = DateTimeField()
  eps = FloatField()
  dt_eps = FloatField()
  total_revenue_ps = FloatField()
  revenue_ps = FloatField()
  capital_rese_ps = FloatField()
  surplus_rese_ps = FloatField()
  undist_profit_ps = FloatField()
  extra_item = FloatField()
  profit_dedt = FloatField()
  gross_margin = FloatField()
  current_ratio = FloatField()
  quick_ratio = FloatField()
  cash_ratio = FloatField()
  ar_turn = FloatField()
  ca_turn = FloatField()
  fa_turn = FloatField()
  assets_turn = FloatField()
  op_income = FloatField()
  ebit = FloatField()
  ebitda = FloatField()
  fcff = FloatField()
  fcfe = FloatField()
  current_exint = FloatField()
  noncurrent_exint = FloatField()
  interestdebt = FloatField()
  netdebt = FloatField()
  tangible_asset = FloatField()
  working_capital = FloatField()
  networking_capital = FloatField()
  invest_capital = FloatField()
  retained_earnings = FloatField()
  diluted2_eps = FloatField()
  bps = FloatField()
  ocfps = FloatField()
  retainedps = FloatField()
  cfps = FloatField()
  ebit_ps = FloatField()
  fcff_ps = FloatField()
  fcfe_ps = FloatField()
  netprofit_margin = FloatField()
  grossprofit_margin = FloatField()
  cogs_of_sales = FloatField()
  expense_of_sales = FloatField()
  profit_to_gr = FloatField()
  saleexp_to_gr = FloatField()
  adminexp_of_gr = FloatField()
  finaexp_of_gr = FloatField()
  impai_ttm = FloatField()
  gc_of_gr = FloatField()
  op_of_gr = FloatField()
  ebit_of_gr = FloatField()
  roe = FloatField()
  roe_waa = FloatField()
  roe_dt = FloatField()
  roa = FloatField()
  npta = FloatField()
  roic = FloatField()
  roe_yearly = FloatField()
  roa2_yearly = FloatField()
  debt_to_assets = FloatField()
  assets_to_eqt = FloatField()
  dp_assets_to_eqt = FloatField()
  ca_to_assets = FloatField()
  nca_to_assets = FloatField()
  tbassets_to_totalassets = FloatField()
  int_to_talcap = FloatField()
  eqt_to_talcapital = FloatField()
  currentdebt_to_debt = FloatField()
  longdeb_to_debt = FloatField()
  ocf_to_shortdebt = FloatField()
  debt_to_eqt = FloatField()
  eqt_to_debt = FloatField()
  eqt_to_interestdebt = FloatField()
  tangibleasset_to_debt = FloatField()
  tangasset_to_intdebt = FloatField()
  tangibleasset_to_netdebt = FloatField()
  ocf_to_debt = FloatField()
  turn_days = FloatField()
  roa_yearly = FloatField()
  roa_dp = FloatField()
  fixed_assets = FloatField()
  profit_to_op = FloatField()
  q_saleexp_to_gr = FloatField()
  q_gc_to_gr = FloatField()
  q_roe = FloatField()
  q_dt_roe = FloatField()
  q_npta = FloatField()
  q_ocf_to_sales = FloatField()
  basic_eps_yoy = FloatField()
  dt_eps_yoy = FloatField()
  cfps_yoy = FloatField()
  op_yoy = FloatField()
  ebt_yoy = FloatField()
  netprofit_yoy = FloatField()
  dt_netprofit_yoy = FloatField()
  ocf_yoy = FloatField()
  roe_yoy = FloatField()
  bps_yoy = FloatField()
  assets_yoy = FloatField()
  eqt_yoy = FloatField()
  tr_yoy = FloatField()
  or_yoy = FloatField()
  q_sales_yoy = FloatField()
  q_op_qoq = FloatField()
  equity_yoy = FloatField()

  def __str__(self):
    f = ""
    for k in dir(self):
      if k[0] != '_' and hasattr(self, k):
        v = getattr(self, k)
        if type(v) in [float, str, int, datetime]:
          f = f + f"{k} = {v}\n"
    return f

  def __repr__(self):
    return str(self)


class EquityCashFlowInfo(Document):
  meta = {
    "indexes": [
      {
        "fields": ("ts_code", "end_date"),
        "unique": True,
      }
    ]}
  
  ts_code = StringField()
  ann_date = DateTimeField()
  f_ann_date = DateTimeField()
  end_date = DateTimeField()

  comp_type = StringField()
  report_type = StringField()
  net_profit = FloatField()
  finan_exp = FloatField()
  c_fr_sale_sg = FloatField()
  recp_tax_rends = FloatField()
  n_depos_incr_fi = FloatField()
  n_incr_loans_cb = FloatField()
  n_inc_borr_oth_fi = FloatField()
  prem_fr_orig_contr = FloatField()
  n_incr_insured_dep = FloatField()
  n_reinsur_prem = FloatField()
  n_incr_disp_tfa = FloatField()
  ifc_cash_incr = FloatField()
  n_incr_disp_faas = FloatField()
  n_incr_loans_oth_bank = FloatField()
  n_cap_incr_repur = FloatField()
  c_fr_oth_operate_a = FloatField()
  c_inf_fr_operate_a = FloatField()
  c_paid_goods_s = FloatField()
  c_paid_to_for_empl = FloatField()
  c_paid_for_taxes = FloatField()
  n_incr_clt_loan_adv = FloatField()
  n_incr_dep_cbob = FloatField()
  c_pay_claims_orig_inco = FloatField()
  pay_handling_chrg = FloatField()
  pay_comm_insur_plcy = FloatField()
  oth_cash_pay_oper_act = FloatField()
  st_cash_out_act = FloatField()
  n_cashflow_act = FloatField()
  oth_recp_ral_inv_act = FloatField()
  c_disp_withdrwl_invest = FloatField()
  c_recp_return_invest = FloatField()
  n_recp_disp_fiolta = FloatField()
  n_recp_disp_sobu = FloatField()
  stot_inflows_inv_act = FloatField()
  c_pay_acq_const_fiolta = FloatField()
  c_paid_invest = FloatField()
  n_disp_subs_oth_biz = FloatField()
  oth_pay_ral_inv_act = FloatField()
  n_incr_pledge_loan = FloatField()
  stot_out_inv_act = FloatField()
  n_cashflow_inv_act = FloatField()
  c_recp_borrow = FloatField()
  proc_issue_bonds = FloatField()
  oth_cash_recp_ral_fnc_act = FloatField()
  stot_cash_in_fnc_act = FloatField()
  free_cashflow = FloatField()
  c_prepay_amt_borr = FloatField()
  c_pay_dist_dpcp_int_exp = FloatField()
  incl_dvd_profit_paid_sc_ms = FloatField()
  oth_cashpay_ral_fnc_act = FloatField()
  stot_cashout_fnc_act = FloatField()
  n_cash_flows_fnc_act = FloatField()
  eff_fx_flu_cash = FloatField()
  n_incr_cash_cash_equ = FloatField()
  c_cash_equ_beg_period = FloatField()
  c_cash_equ_end_period = FloatField()
  c_recp_cap_contrib = FloatField()
  incl_cash_rec_saims = FloatField()
  uncon_invest_loss = FloatField()
  prov_depr_assets = FloatField()
  depr_fa_coga_dpba = FloatField()
  amort_intang_assets = FloatField()
  lt_amort_deferred_exp = FloatField()
  decr_deferred_exp = FloatField()
  incr_acc_exp = FloatField()
  loss_disp_fiolta = FloatField()
  loss_scr_fa = FloatField()
  loss_fv_chg = FloatField()
  invest_loss = FloatField()
  decr_def_inc_tax_assets = FloatField()
  incr_def_inc_tax_liab = FloatField()
  decr_inventories = FloatField()
  decr_oper_payable = FloatField()
  incr_oper_payable = FloatField()
  others = FloatField()
  im_net_cashflow_oper_act = FloatField()
  conv_debt_into_cap = FloatField()
  conv_copbonds_due_within_1y = FloatField()
  fa_fnc_leases = FloatField()
  end_bal_cash = FloatField()
  beg_bal_cash = FloatField()
  end_bal_cash_equ = FloatField()
  beg_bal_cash_equ = FloatField()
  im_n_incr_cash_equ = FloatField()

  
  def __str__(self):
    f = ""
    for k in dir(self):
      if k[0] != '_' and hasattr(self, k):
        v = getattr(self, k)
        if type(v) in [float, str, int, datetime]:
          f = f + f"{k} = {v}\n"
    return f

  def __repr__(self):
    return str(self)


class EquityBalanceSheetInfo(Document):
  meta = {
    "indexes": [
      {
        "fields": ("ts_code", "end_date"),
        "unique": True,
      }
    ]}

  ts_code = StringField()
  ann_date = DateTimeField()
  f_ann_date = DateTimeField()
  end_date = DateTimeField()
  report_type = StringField()
  comp_type = StringField()
  total_share = FloatField()
  cap_rese = FloatField()
  undistr_porfit = FloatField()
  surplus_rese = FloatField()
  special_rese = FloatField()
  money_cap = FloatField()
  trad_asset = FloatField()
  notes_receiv = FloatField()
  accounts_receiv = FloatField()
  oth_receiv = FloatField()
  prepayment = FloatField()
  div_receiv = FloatField()
  int_receiv = FloatField()
  inventories = FloatField()
  amor_exp = FloatField()
  nca_within_1y = FloatField()
  sett_rsrv = FloatField()
  loanto_oth_bank_fi = FloatField()
  premium_receiv = FloatField()
  reinsur_receiv = FloatField()
  reinsur_res_receiv = FloatField()
  pur_resale_fa = FloatField()
  oth_cur_assets = FloatField()
  total_cur_assets = FloatField()
  fa_avail_for_sale = FloatField()
  htm_invest = FloatField()
  lt_eqt_invest = FloatField()
  invest_real_estate = FloatField()
  time_deposits = FloatField()
  oth_assets = FloatField()
  lt_rec = FloatField()
  fix_assets = FloatField()
  cip = FloatField()
  const_materials = FloatField()
  fixed_assets_disp = FloatField()
  produc_bio_assets = FloatField()
  oil_and_gas_assets = FloatField()
  intan_assets = FloatField()
  r_and_d = FloatField()
  goodwill = FloatField()
  lt_amor_exp = FloatField()
  defer_tax_assets = FloatField()
  decr_in_disbur = FloatField()
  oth_nca = FloatField()
  total_nca = FloatField()
  cash_reser_cb = FloatField()
  depos_in_oth_bfi = FloatField()
  prec_metals = FloatField()
  deriv_assets = FloatField()
  rr_reins_une_prem = FloatField()
  rr_reins_outstd_cla = FloatField()
  rr_reins_lins_liab = FloatField()
  rr_reins_lthins_liab = FloatField()
  refund_depos = FloatField()
  ph_pledge_loans = FloatField()
  refund_cap_depos = FloatField()
  indep_acct_assets = FloatField()
  client_depos = FloatField()
  client_prov = FloatField()
  transac_seat_fee = FloatField()
  invest_as_receiv = FloatField()
  total_assets = FloatField()
  lt_borr = FloatField()
  st_borr = FloatField()
  cb_borr = FloatField()
  depos_ib_deposits = FloatField()
  loan_oth_bank = FloatField()
  trading_fl = FloatField()
  notes_payable = FloatField()
  acct_payable = FloatField()
  adv_receipts = FloatField()
  sold_for_repur_fa = FloatField()
  comm_payable = FloatField()
  payroll_payable = FloatField()
  taxes_payable = FloatField()
  int_payable = FloatField()
  div_payable = FloatField()
  oth_payable = FloatField()
  acc_exp = FloatField()
  deferred_inc = FloatField()
  st_bonds_payable = FloatField()
  payable_to_reinsurer = FloatField()
  rsrv_insur_cont = FloatField()
  acting_trading_sec = FloatField()
  acting_uw_sec = FloatField()
  non_cur_liab_due_1y = FloatField()
  oth_cur_liab = FloatField()
  total_cur_liab = FloatField()
  bond_payable = FloatField()
  lt_payable = FloatField()
  specific_payables = FloatField()
  estimated_liab = FloatField()
  defer_tax_liab = FloatField()
  defer_inc_non_cur_liab = FloatField()
  oth_ncl = FloatField()
  total_ncl = FloatField()
  depos_oth_bfi = FloatField()
  deriv_liab = FloatField()
  depos = FloatField()
  agency_bus_liab = FloatField()
  oth_liab = FloatField()
  prem_receiv_adva = FloatField()
  depos_received = FloatField()
  ph_invest = FloatField()
  reser_une_prem = FloatField()
  reser_outstd_claims = FloatField()
  reser_lins_liab = FloatField()
  reser_lthins_liab = FloatField()
  indept_acc_liab = FloatField()
  pledge_borr = FloatField()
  indem_payable = FloatField()
  policy_div_payable = FloatField()
  total_liab = FloatField()
  treasury_share = FloatField()
  ordin_risk_reser = FloatField()
  forex_differ = FloatField()
  invest_loss_unconf = FloatField()
  minority_int = FloatField()
  total_hldr_eqy_exc_min_int = FloatField()
  total_hldr_eqy_inc_min_int = FloatField()
  total_liab_hldr_eqy = FloatField()
  lt_payroll_payable = FloatField()
  oth_comp_income = FloatField()
  oth_eqt_tools = FloatField()
  oth_eqt_tools_p_shr = FloatField()
  lending_funds = FloatField()
  acc_receivable = FloatField()
  st_fin_payable = FloatField()
  payables = FloatField()
  hfs_assets = FloatField()
  hfs_sales = FloatField()

  def __str__(self):
    f = ""
    for k in dir(self):
      if k[0] != '_' and hasattr(self, k):
        v = getattr(self, k)
        if type(v) in [float, str, int, datetime]:
          f = f + f"{k} = {v}\n"
    return f

  def __repr__(self):
    return str(self)
    
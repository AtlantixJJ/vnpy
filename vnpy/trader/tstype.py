from mongoengine import DateTimeField, Document, IntField, FloatField, StringField
import numpy as np
from datetime import datetime


def get_key_type(df):
  dic = {}
  for k in df.keys():
    t = "StringField()"
    if type(df[k][0]) is np.int64:
      t = "IntField()"
    elif type(df[k][0]) is np.float64:
      t = "FloatField()"
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
  prem_earned = StringField()
  comm_income = FloatField()
  n_commis_income = FloatField()
  n_oth_income = FloatField()
  n_oth_b_income = FloatField()
  prem_income = StringField()
  out_prem = StringField()
  une_prem_reser = StringField()
  reins_income = StringField()
  n_sec_tb_income = StringField()
  n_sec_uw_income = StringField()
  n_asset_mg_income = StringField()
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
  prem_refund = StringField()
  compens_payout = StringField()
  reser_insur_liab = StringField()
  div_payt = StringField()
  reins_exp = StringField()
  oper_exp = FloatField()
  compens_payout_refu = StringField()
  insur_reser_refu = StringField()
  reins_cost_refund = StringField()
  other_bus_cost = FloatField()
  operate_profit = FloatField()
  non_oper_income = FloatField()
  non_oper_exp = FloatField()
  nca_disploss = StringField()
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
  insurance_exp = StringField()
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
  gross_margin = StringField()
  current_ratio = FloatField()
  quick_ratio = FloatField()
  cash_ratio = StringField()
  ar_turn = StringField()
  ca_turn = StringField()
  fa_turn = FloatField()
  assets_turn = FloatField()
  op_income = FloatField()
  ebit = StringField()
  ebitda = StringField()
  fcff = FloatField()
  fcfe = StringField()
  current_exint = StringField()
  noncurrent_exint = StringField()
  interestdebt = StringField()
  netdebt = StringField()
  tangible_asset = StringField()
  working_capital = StringField()
  networking_capital = StringField()
  invest_capital = StringField()
  retained_earnings = FloatField()
  diluted2_eps = FloatField()
  bps = FloatField()
  ocfps = FloatField()
  retainedps = FloatField()
  cfps = FloatField()
  ebit_ps = StringField()
  fcff_ps = StringField()
  fcfe_ps = StringField()
  netprofit_margin = FloatField()
  grossprofit_margin = StringField()
  cogs_of_sales = StringField()
  expense_of_sales = StringField()
  profit_to_gr = FloatField()
  saleexp_to_gr = StringField()
  adminexp_of_gr = FloatField()
  finaexp_of_gr = StringField()
  impai_ttm = FloatField()
  gc_of_gr = StringField()
  op_of_gr = FloatField()
  ebit_of_gr = StringField()
  roe = FloatField()
  roe_waa = FloatField()
  roe_dt = FloatField()
  roa = StringField()
  npta = FloatField()
  roic = StringField()
  roe_yearly = FloatField()
  roa2_yearly = StringField()
  debt_to_assets = FloatField()
  assets_to_eqt = FloatField()
  dp_assets_to_eqt = FloatField()
  ca_to_assets = StringField()
  nca_to_assets = StringField()
  tbassets_to_totalassets = StringField()
  int_to_talcap = StringField()
  eqt_to_talcapital = StringField()
  currentdebt_to_debt = StringField()
  longdeb_to_debt = StringField()
  ocf_to_shortdebt = StringField()
  debt_to_eqt = FloatField()
  eqt_to_debt = FloatField()
  eqt_to_interestdebt = StringField()
  tangibleasset_to_debt = StringField()
  tangasset_to_intdebt = StringField()
  tangibleasset_to_netdebt = StringField()
  ocf_to_debt = FloatField()
  turn_days = StringField()
  roa_yearly = FloatField()
  roa_dp = FloatField()
  fixed_assets = FloatField()
  profit_to_op = FloatField()
  q_saleexp_to_gr = StringField()
  q_gc_to_gr = StringField()
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
  finan_exp = StringField()
  c_fr_sale_sg = StringField()
  recp_tax_rends = StringField()
  n_depos_incr_fi = FloatField()
  n_incr_loans_cb = FloatField()
  n_inc_borr_oth_fi = FloatField()
  prem_fr_orig_contr = StringField()
  n_incr_insured_dep = StringField()
  n_reinsur_prem = StringField()
  n_incr_disp_tfa = StringField()
  ifc_cash_incr = FloatField()
  n_incr_disp_faas = StringField()
  n_incr_loans_oth_bank = FloatField()
  n_cap_incr_repur = FloatField()
  c_fr_oth_operate_a = FloatField()
  c_inf_fr_operate_a = FloatField()
  c_paid_goods_s = StringField()
  c_paid_to_for_empl = FloatField()
  c_paid_for_taxes = FloatField()
  n_incr_clt_loan_adv = FloatField()
  n_incr_dep_cbob = FloatField()
  c_pay_claims_orig_inco = StringField()
  pay_handling_chrg = FloatField()
  pay_comm_insur_plcy = StringField()
  oth_cash_pay_oper_act = FloatField()
  st_cash_out_act = FloatField()
  n_cashflow_act = FloatField()
  oth_recp_ral_inv_act = FloatField()
  c_disp_withdrwl_invest = FloatField()
  c_recp_return_invest = FloatField()
  n_recp_disp_fiolta = FloatField()
  n_recp_disp_sobu = StringField()
  stot_inflows_inv_act = FloatField()
  c_pay_acq_const_fiolta = FloatField()
  c_paid_invest = FloatField()
  n_disp_subs_oth_biz = StringField()
  oth_pay_ral_inv_act = FloatField()
  n_incr_pledge_loan = StringField()
  stot_out_inv_act = FloatField()
  n_cashflow_inv_act = FloatField()
  c_recp_borrow = StringField()
  proc_issue_bonds = FloatField()
  oth_cash_recp_ral_fnc_act = FloatField()
  stot_cash_in_fnc_act = FloatField()
  free_cashflow = FloatField()
  c_prepay_amt_borr = FloatField()
  c_pay_dist_dpcp_int_exp = FloatField()
  incl_dvd_profit_paid_sc_ms = StringField()
  oth_cashpay_ral_fnc_act = StringField()
  stot_cashout_fnc_act = FloatField()
  n_cash_flows_fnc_act = FloatField()
  eff_fx_flu_cash = FloatField()
  n_incr_cash_cash_equ = FloatField()
  c_cash_equ_beg_period = FloatField()
  c_cash_equ_end_period = FloatField()
  c_recp_cap_contrib = StringField()
  incl_cash_rec_saims = StringField()
  uncon_invest_loss = StringField()
  prov_depr_assets = FloatField()
  depr_fa_coga_dpba = FloatField()
  amort_intang_assets = FloatField()
  lt_amort_deferred_exp = FloatField()
  decr_deferred_exp = FloatField()
  incr_acc_exp = FloatField()
  loss_disp_fiolta = FloatField()
  loss_scr_fa = StringField()
  loss_fv_chg = FloatField()
  invest_loss = FloatField()
  decr_def_inc_tax_assets = FloatField()
  incr_def_inc_tax_liab = FloatField()
  decr_inventories = StringField()
  decr_oper_payable = FloatField()
  incr_oper_payable = FloatField()
  others = FloatField()
  im_net_cashflow_oper_act = FloatField()
  conv_debt_into_cap = StringField()
  conv_copbonds_due_within_1y = StringField()
  fa_fnc_leases = StringField()
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
  ann_date = StringField()
  f_ann_date = StringField()
  end_date = StringField()
  report_type = StringField()
  comp_type = StringField()
  total_share = FloatField()
  cap_rese = FloatField()
  undistr_porfit = FloatField()
  surplus_rese = FloatField()
  special_rese = StringField()
  money_cap = FloatField()
  trad_asset = FloatField()
  notes_receiv = FloatField()
  accounts_receiv = FloatField()
  oth_receiv = FloatField()
  prepayment = FloatField()
  div_receiv = StringField()
  int_receiv = FloatField()
  inventories = FloatField()
  amor_exp = FloatField()
  nca_within_1y = StringField()
  sett_rsrv = StringField()
  loanto_oth_bank_fi = FloatField()
  premium_receiv = StringField()
  reinsur_receiv = StringField()
  reinsur_res_receiv = StringField()
  pur_resale_fa = FloatField()
  oth_cur_assets = StringField()
  total_cur_assets = FloatField()
  fa_avail_for_sale = FloatField()
  htm_invest = FloatField()
  lt_eqt_invest = FloatField()
  invest_real_estate = FloatField()
  time_deposits = StringField()
  oth_assets = FloatField()
  lt_rec = StringField()
  fix_assets = FloatField()
  cip = FloatField()
  const_materials = StringField()
  fixed_assets_disp = FloatField()
  produc_bio_assets = StringField()
  oil_and_gas_assets = StringField()
  intan_assets = FloatField()
  r_and_d = StringField()
  goodwill = FloatField()
  lt_amor_exp = FloatField()
  defer_tax_assets = FloatField()
  decr_in_disbur = FloatField()
  oth_nca = StringField()
  total_nca = FloatField()
  cash_reser_cb = FloatField()
  depos_in_oth_bfi = FloatField()
  prec_metals = FloatField()
  deriv_assets = FloatField()
  rr_reins_une_prem = StringField()
  rr_reins_outstd_cla = StringField()
  rr_reins_lins_liab = StringField()
  rr_reins_lthins_liab = StringField()
  refund_depos = StringField()
  ph_pledge_loans = StringField()
  refund_cap_depos = StringField()
  indep_acct_assets = StringField()
  client_depos = StringField()
  client_prov = StringField()
  transac_seat_fee = StringField()
  invest_as_receiv = FloatField()
  total_assets = FloatField()
  lt_borr = FloatField()
  st_borr = FloatField()
  cb_borr = FloatField()
  depos_ib_deposits = StringField()
  loan_oth_bank = FloatField()
  trading_fl = FloatField()
  notes_payable = StringField()
  acct_payable = FloatField()
  adv_receipts = FloatField()
  sold_for_repur_fa = FloatField()
  comm_payable = StringField()
  payroll_payable = FloatField()
  taxes_payable = FloatField()
  int_payable = FloatField()
  div_payable = FloatField()
  oth_payable = FloatField()
  acc_exp = FloatField()
  deferred_inc = FloatField()
  st_bonds_payable = StringField()
  payable_to_reinsurer = StringField()
  rsrv_insur_cont = StringField()
  acting_trading_sec = StringField()
  acting_uw_sec = StringField()
  non_cur_liab_due_1y = FloatField()
  oth_cur_liab = FloatField()
  total_cur_liab = FloatField()
  bond_payable = FloatField()
  lt_payable = FloatField()
  specific_payables = StringField()
  estimated_liab = FloatField()
  defer_tax_liab = FloatField()
  defer_inc_non_cur_liab = StringField()
  oth_ncl = StringField()
  total_ncl = FloatField()
  depos_oth_bfi = FloatField()
  deriv_liab = FloatField()
  depos = FloatField()
  agency_bus_liab = FloatField()
  oth_liab = FloatField()
  prem_receiv_adva = StringField()
  depos_received = StringField()
  ph_invest = StringField()
  reser_une_prem = StringField()
  reser_outstd_claims = StringField()
  reser_lins_liab = StringField()
  reser_lthins_liab = StringField()
  indept_acc_liab = StringField()
  pledge_borr = StringField()
  indem_payable = StringField()
  policy_div_payable = StringField()
  total_liab = FloatField()
  treasury_share = StringField()
  ordin_risk_reser = FloatField()
  forex_differ = FloatField()
  invest_loss_unconf = StringField()
  minority_int = FloatField()
  total_hldr_eqy_exc_min_int = FloatField()
  total_hldr_eqy_inc_min_int = FloatField()
  total_liab_hldr_eqy = FloatField()
  lt_payroll_payable = StringField()
  oth_comp_income = FloatField()
  oth_eqt_tools = FloatField()
  oth_eqt_tools_p_shr = FloatField()
  lending_funds = StringField()
  acc_receivable = StringField()
  st_fin_payable = StringField()
  payables = StringField()
  hfs_assets = StringField()
  hfs_sales = StringField()

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
    
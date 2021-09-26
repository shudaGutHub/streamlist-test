import pandas as pd
import numpy as np
import pathlib
import scipy.stats as stats
import pathlib as pathlib
from mibian import BS
import yfinance as yf
from helpers import convert_dates_YYYY_mm_dd, convert_to_USD, force_float
#from BDFunds import *

TARGET_EXPIRY = '2021-12-17'


fp_DashSummary = pathlib.Path( "Z:\\rshinydata\\summary\\DashSummary.csv")
fp_BDFund = pathlib.Path("Z:\\rshinydata\\summary\\BDFundPortfolio.csv")
fp_TradeList = pathlib.Path("Z:\\rshinydata\\summary\\SampleTradeList.csv")

path_dash = pathlib.Path("Z:\\rshinydata\\summary")
path_options = pathlib.Path("Z:\\rshinydata\\currentoptdata")
path_equities = pathlib.Path("Z:\\rshinydata\\pricedata")
path_options_history = pathlib.Path("Z:\\rshinydata\\pricedata")

options_bad_files = ['CurrentZones - Copy.csv', 'CurrentZones.csv', '_QuoteSummary.csv']

DIR_DATA = pathlib.Path('C:\\Users\\salee\\projects\\streamlit-example')


TABLE_HOLDINGS_END = 'BDIN_HOLDINGS_20210913'
filename_tickers = "tickers.csv"
filename_attribution_ytd = "BDIN_Security_YTD.xlsx"
filename_attribution_ltd = "BDIN_Security_LTMONTH.xlsx"
filename_holdings_end = "BDIN__HOLDINGS_2021_Aug.xlsx"
filename_trades = "BDIN_TRADES_SI_20210916.xlsx"#BDIN_Trades_20210916.xlsx'
filename_NAV = 'BDIN_NAV.xlsx'

get_fp = lambda fname: pathlib.Path(DIR_DATA,fname)


def get_yahoo_symbols():
    """Map from ISIN to yahoo tickers"""
    ISIN = {'US00183L1026': 'ANGI',
 'US2383371091': 'PLAY',
 'FR0000120404': 'AC.PA',
 'US2220702037': 'COTY',
 'CH0023405456': 'DUFN.SW',
 'US9831341071': 'WYNN',
 'US01609W1027': 'BABA',
 'MXP001391012': 'ALSEA.MX',
 'US88339P1012': 'REAL',
 'US4824971042': 'BEKE',
 'US8740801043': 'TAL',
 'IT0003506190': 'ATL.MI',
 'DE0007500001': 'TKA.DE',
 'US55826T1025': 'MSGE',
 'ES0109067019': 'AMS.MC',
 'US88032Q1094': 'TCEHY',
 'NL0013654783': 'PROSF',
 'BRCIELACNOR3': 'CIEL3.SA',
 'US4500561067': 'IRTC',
 'US78573M1045': 'SABR',
 'US49639K1016': 'KC',
 'US9851941099': 'YSG',
 'US05278C1071': 'ATHM',
 'ES0176252718': 'MEL.MC',
 'GB0004762810': 'JSG.L',
 'GB00BGBN7C04': 'SSPG.L',
 'IT0001137345': 'AGL.MI',
 'US6475811070': 'EDU',
 'KYG6470A1168': '9901.HK',
 'ES0105046009': 'AENA.MC',
 'DE000A0D9PT0': 'MTX.DE',
 'BABAL180U': 'BABA',
 'SPYU430U': 'SPY',
 'BIDUL170U': 'BIDU',
 'DE0006335003': 'KRN.DE',
 'BEKEV15U': 'BEKE',
 'CNE100003688': '0788.HK',
 'WYNNO70U': 'WYNN',
 'GB00B15FWH70': 'CINE.L',
 'SPY2U425U': 'SPY',
 'TALK325U': 'TAL',
 'BABAI230U': 'BABA',
 'US6549022043': 'NOK',
 'BABAI225U': 'BABA',
 'SPY2U400U': 'SPY',
 'WYNNO50U': 'WYNN',
 'SPYU410U': 'SPY',
 'TALU75U': 'TAL',
 'EDUU45U': 'EDU'}
    return ISIN

s3bucket = "s3://blkd/rshinydata/pricedata/"
fps3_BDFund = "s3://blkd/rshinydata/summary/BDFundPortfolio.csv"  # Positions data derived from .xlsx file
fps3_TradeList = "s3://blkd/rshinydata/summary/SampleTradeList.csv"  # Generated for only call options
fps3_DashSummary = "s3://blkd/rshinydata/summary/DashSummary.csv"
dfds = read_s3_csv(fps3_DashSummary) #DashSummary all calls
dfds['Currency'] = dfds['Currency'].str.replace("GBp","GBP")
dfds['YahooTicker'] = dfds['Symbol'].str.replace("0700.HK","TCEHY")
dfds_GLOB = dfds[~dfds.Currency.isin(["USD"])].set_index('YahooTicker')[['Currency', 'ExchangeRate']]



yahootickers_file = "C://Users//Saleem/projects/X8MKB/data/MAP_YAHOO_TICKERS.CSV"
dftickers = pd.read_csv(yahootickers_file)  # ,index_col=['Ticker'], usecols=['Ticker','YahooTicker'])

TARGET_EXPIRY = input("TARGET_EXPIRY:")
s3bucket = "s3://blkd/rshinydata/pricedata/"
fps3_BDFund = "s3://blkd/rshinydata/summary/BDFundPortfolio.csv"
df_BDFund = read_s3_csv(fp=fps3_BDFund)


def load_trades_db(table, path_db):
    """Loads TrueQuant exports from database
    fund: "BDIN"
    conn: conn = sqlite3.connect("bdin.db")
    trades_BDIN
    trades_BDEQ
    trades_BDOP"""

    conn = sqlite3.connect(path_db)
    table = table

    data = pd.read_sql(con=conn, sql=f"SELECT * FROM {table}")
    data.columns = [c.replace("/", "").strip().replace(" ", "").replace("%", "Pct") for c in data.columns]
    data['TradeData'] = helpers.convert_dates_YYYY_mm_dd(data, 'TradeData')
    data['EffectiveDate'] = helpers.convert_dates_YYYY_mm_dd(data, 'EffectiveDate')
    data['Date'] = helpers.convert_dates_YYYY_mm_dd(data, 'TradeData')
    data['ValueTrade'] = helpers.force_float(data, 'Value')
    data['SharesPar'] = helpers.force_float(data, 'SharesPar')
    data['PriceBase'] = helpers.force_float(data, 'PriceBase')
    data['PriceLocal'] = helpers.force_float(data, 'PriceLocal')
    data['FirstTradeDate'] = data.groupby('Ticker')['TradeData'].transform(lambda d:d.min())

    data = add_risk_multiplier(data)
    df = process_trades(data)
    return df


def process_trades(df, holdings_start=None):
    """Clean columns"""
    df.columns = [c.replace("/", "").strip().replace(" ", "").replace("%", "Pct") for c in df.columns]

    dfeq = df.query("AssetClass==['Equity','Option']").copy()
    dfeq['Symbol'] = dfeq['Ticker'].str.split(" ").map(lambda x: x[0])

    option_tickers_raw = dfeq.query('AssetClass == "Option"')['Ticker'].unique()

    options_split = {ticker: dict(zip(["Symbol", "Currency", "ExpiryDate", "PCStrike"], ticker.split(" "))) for ticker
                     in option_tickers_raw}
    options_split_Expiry = {ticker: pd.to_datetime(opt.get('ExpiryDate')) for ticker, opt in options_split.items()}

    options_split_PutCall = {ticker: opt.get('PCStrike')[0] for ticker, opt in options_split.items()}
    options_split_Strike = {ticker: float(opt.get('PCStrike')[1:]) for ticker, opt in options_split.items()}

    dfeq['OptPC'] = dfeq['Ticker'].map(lambda x: options_split_PutCall.get(x, "C"))

    dfeq['UNDERLYING_LAST_PRICE'] = dfeq['PriceBase']
    dfeq['STRIKE'] = dfeq['Ticker'].map(
        lambda x: options_split_Strike.get(x, .01))  # StrikePrice or 1 cent for equities
    dfeq['EXPIRY_DATE'] = dfeq['Ticker'].map(lambda x: options_split_Expiry.get(x, pd.to_datetime(
        '2050-01-01')))  # Expiry or some date far in the future for Equity
    dfeq['TERM_DAYS'] = (dfeq['EXPIRY_DATE'] - dfeq['TradeData']).map(
        lambda x: max(x.days, 0))  # Number of days in Term
    dfeq['TERM_YEARS'] = dfeq['TERM_DAYS'] / 365.0
    dfeq['RATE'] = .01
    dfeq['RATE_Q'] = .0
    dfeq['X8VOL'] = .25
    dfeq['BSVOL'] = dfeq['X8VOL'] * 100
    dfeq['Date'] = dfeq['TradeData'].values

    dfeq = dfeq.set_index(['Date', 'Ticker'])
    return dfeq





map_FX = {'Europe':1.2, 'United States':1, 'Mexico': .25, 'Brazil':.2, 'United Kingdom':1.2, 'Switzerland':1.3,'China':1.0, 'Hong Kong':.6}
class EquityModel(object):
	def __init__(self, sym, ISIN):
		self.sym = sym
		self.ISIN = ISIN
		self.data = pd.DataFrame

	def load(self, data):
		self.data = data

	def price(self, value_date):
		return self.data.loc[value_date]

	def __repr__(self):
		return self.sym


class OptionModel(object):
	def __init__(self, value_date, sym, spot, sigma, optPC, strike, expiry, frate, qrate, bdfunds_ticker=None):
		"""optPC:"C", sym:"CRM", strike:227, expiry"""
		print("Sym: {}".format(sym))
		self.value_date = value_date
		self.sym = sym
		self.spot = spot
		self.sigma = sigma
		self.optPC = optPC
		self.strike = strike
		self.expiry = expiry
		self.frate = frate
		self.qrate = qrate
		self.EXPIRY_YF = self.expiry - (pd.Timedelta(1, 'DAYS'))
		self.bdfunds_ticker = bdfunds_ticker
		self.option_ticker = "_".join([optPC, sym, str(strike), expiry.strftime("%Y%m%d")])
		self.underlying_ticker = yf.Ticker(self.sym)
		self.data_s3 = None
		self.yahoodata = self.get_yahoo()
		self.impliedVol = self.yahoodata['impliedVolatility']

	def add_underlying(self, bdpos):
		try:
			self.underlying = yf.Ticker(self.sym)
		except:
			self.underlying = yf.Ticker(f"input valid underlying {self.sym}")

	# self.underlying = BDEquity(sym, start)
	def load_s3(self):
		self.data_s3 = load_s3_options(self.sym)
	
	def term_in_years(self):
		return (self.expiry - self.value_date).days / 365.0

	def get_yahoo(self, chain=False):
		
		equity = self.underlying_ticker
		test_expiry = self.EXPIRY_YF  # .strftime("%Y-%m-%d")
		# test_expiry not in self.underlying_ticker.option_chain(test_expiry).calls.query("strike==@self.strike") #.strftime("%Y-%m-%d")- (pd.Timedelta(1, 'DAYS'))
		if test_expiry < self.value_date:
			print(f"Option expired {self.option_ticker}")
			return 0.01

		try:
			if self.optPC == "C":
				return self.underlying_ticker.option_chain(test_expiry).calls.query("strike==@self.strike")
			if self.optPC == "P":
				return self.underlying_ticker.option_chain(test_expiry).puts.query("strike==@self.strike")
			else:
				print("Not P or C")
				return None
		except:
			
			expiration_dates = equity.options
			print("Expiration Dates: ", expiration_dates)
			self.expiry = input("Please enter valid date:")
			if self.optPC == "C":
				return self.underlying_ticker.option_chain(test_expiry).calls.query("strike==@self.strike")
			if self.optPC == "P":
				return self.underlying_ticker.option_chain(test_expiry).puts.query("strike==@self.strike")
			else:
				print("Not P or C")
				return None

	@staticmethod
	def bsm_price(option_type, sigma, s, k, r, T, q):
		# calculate the bsm price of European call and put options

		sigma = float(sigma)
		d1 = (np.log(s / k) + (r - q + sigma ** 2 * 0.5) * T) / (sigma * np.sqrt(T))
		d2 = d1 - sigma * np.sqrt(T)
		if option_type == 'C':
			price = np.exp(-r * T) * (s * np.exp((r - q) * T) * stats.norm.cdf(d1) - k * stats.norm.cdf(d2))
			return price
		elif option_type == 'P':
			price = np.exp(-r * T) * (k * stats.norm.cdf(-d2) - s * np.exp((r - q) * T) * stats.norm.cdf(-d1))
			return price
		else:
			print('No such option type %s') % option_type

	# Sharpe Ratio

	def theo_price(self):

		return self.bsm_price(option_type=self.optPC,
							  sigma=self.sigma,
							  s=self.spot,
							  k=self.strike,
							  r=self.frate,
							  T=self.term_in_years(),
							  q=self.qrate)

	def delta(self, S, K, T, r, optPC, q=0, sigma=.3):
		"""BSM delta formulaa"""
		d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
		result = 0
		if optPC == 'C':
			result = stats.norm.cdf(d1, 0.0, 1.0)
		elif optPC == 'P':
			result = stats.norm.cdf(d1, 0.0, 1.0) - 1

		return result

	def __repr__(self):
		return self.option_ticker


def get_equities(df):
	options = {}
	for row in df.itertuples():
		assert row.AssetClass == "Equity"
		options[row.ISIN] = EquityModel(
			sym=row.SymbolBLK,
			ISIN=row.ISIN)

	return options


def load_prices_underlyings(dft, start_date='2017-01-01', end_date='2021-09-15'):
	"""Load price data"""
	symbols_options = list(dft.query('AssetClass==["Option","Equity"]')['Symbol'].unique())
	prices={}
	for sym in symbols_options:
		prices[sym]= yf.download(sym, start_date, end_date)
	return pd.concat(prices,axis=1)

def get_returns_from_prices(prices):
	return qs.utils.to_log_returns(prices)


def get_volatilities_from_returns(returns):
	return qs.stats.volatility(returns)



def test_load_trades(filename):
	df = pd.read_excel(
	io = get_fp(filename),
	engine = 'openpyxl',
	sheet_name = 'Sheet1',
	skiprows = 0,  # TODO Process from raw file requires stripping top row
	usecols = 'A:M',
	parse_dates = ['Trade Data', 'Effective Date'],

	)
	df.columns = [c.replace("/", "").strip().replace(" ", "").replace("%", "Pct") for c in df.columns]

	### Isolate Equity and Option rows
	dfeq = df.query("AssetClass==['Equity','Option']").copy()
	dfeq['Symbol'] = dfeq['Ticker'].str.split(" ").map(lambda x: x[0])

	### Map to yahoo symbols used on dashboard by using the first word of
	### "SecurityDescription" in trade data
	### and the first word of "LongName" in database, we have already converted to FirstWordSecurity
	#TODO : Clean up this
	dfeq['FirstWordSecurity'] = dfeq['SecurityDescription'].map(lambda x:x.split(" ")[0])
	set_symbols = lambda df: np.where(df['Symbol_DB'].isnull(), df['Symbol'], df['Symbol_DB'])
	dfdash = load_symbols_from_dash() # Get symbols from dfdash

	dfeq = pd.merge(dfeq, dfdash[['FirstWordSecurity', 'Symbol']], left_on='FirstWordSecurity',
						 right_on='FirstWordSecurity', how='left', suffixes=("", "_DB"), indicator=True)
	dfeq['Symbol'] = set_symbols(dfeq)



	option_tickers_raw = dfeq.query('AssetClass == "Option"')['Ticker'].unique()

	options_split = {ticker: dict(zip(["Symbol","Currency","ExpiryDate","PCStrike"],ticker.split(" "))) for ticker in option_tickers_raw}
	options_split_Expiry = {ticker: pd.to_datetime(opt.get('ExpiryDate')) for ticker,opt in options_split.items()}

	options_split_PutCall = {ticker:opt.get('PCStrike')[0] for ticker,opt in options_split.items()}
	options_split_Strike = {ticker: float(opt.get('PCStrike')[1:]) for ticker, opt in options_split.items()}

	dfeq['OptPC'] = dfeq['Ticker'].map(lambda x:options_split_PutCall.get(x,"C"))
	dfeq['STRIKE'] = dfeq['Ticker'].map(lambda x: options_split_Strike.get(x, .01)) #StrikePrice or 1 cent for equities
	dfeq['EXPIRY_DATE'] = dfeq['Ticker'].map(lambda x: options_split_Expiry.get(x,pd.to_datetime('2050-01-01'))) #Expiry or some date far in the future for Equity
	dfeq['TERM_DAYS'] = (dfeq['EXPIRY_DATE'] - dfeq['TradeData']).map(lambda x: max(x.days, 0)) #Number of days in Term
	dfeq['TERM_YEARS'] = dfeq['TERM_DAYS']/365.0
	dfeq['RATE'] = .01
	dfeq['RATE_Q'] =.0
	dfeq['X8VOL'] = .25
	dfeq['BSVOL'] = dfeq['X8VOL']*100
	dfeq['Date'] = dfeq['TradeData'].values
	dfeq = dfeq.set_index(['Date','Ticker'])
	return dfeq

def test_load_holdings_start(filename):
	df = pd.read_excel(
		io=get_fp(filename),
		engine='openpyxl',
		sheet_name='Sheet1',
		skiprows=0,
		usecols='A:W',
	)
	df.columns =[c.replace("/","").strip().replace(" ","").replace("%","Pct") for c in df.columns]

	return df
def test_load_attribution(filename):
	df = pd.read_excel(
		io=get_fp(filename),
		engine='openpyxl',
		sheet_name='Sheet1',
		skiprows=0,
		usecols='A:I'

	)
	df.columns =[c.replace("/","").strip().replace(" ","").replace("%","PctOf") for c in df.columns]

	return df

from datetime import date


import dataclasses
from dataclasses import dataclass
from dataclasses import dataclass
from math import asin, cos, radians, sin, sqrt




def merge_underlyings(dft, prices, vols, col='Close'):
	dfcloses = pd.DataFrame(prices['Close'].stack(), columns=['Close'])
	dfcloses.index.names=['TradeData','Symbol']

	dfm = dft.set_index(['TradeData','Symbol']).join(dfcloses, lsuffix="_t").reset_index()
	vols =pd.Series(vols.stack(),name='HVOL')
	vols.index.names=['TradeData','Ticker']

	dfv = pd.merge(dfm, pd.DataFrame(vols.reset_index()), left_index=True, right_index=True)
	return dfv



def get_options(df, attr_price='PriceBase', attr_vol="HVOL",  start='2017-01-01', end='2021-09-14'):
	dict_impliedVol = {}
	dict_histVol = {}
	option_models = {}
	for row in df.itertuples():
		assert row.AssetClass == "Option"

		sym = row.Symbol
		sigma = row.HVOL

		spot = row.Close
		trade_date = row[0][0] #TradeData (date)
		ticker = row[0][1]   #Ticker
		value_date = trade_date
		print(value_date)
		expiry = row.EXPIRY_DATE
		number_of_days = (expiry - value_date).days
		assert expiry >= value_date

		strike = row.STRIKE
		frate = row.RATE
		qrate = row.RATE_Q
		optPC = row.OptPC
		bdfunds_ticker = None
		option_models[ticker] = {'Symbol': row.Symbol,
								 'ExpiryDate': row.EXPIRY_DATE,
								 'Strike': row.STRIKE,
								 'OptPC': row.OptPC}

		bsmodel = lambda row: row[['Close','Strike','RATE','TERM_DAYS']]
		bsdata = [row.Close,
				  row.STRIKE, #This is a fixed number for the life of the contract
				  row.RATE * 100,
				  number_of_days]

		print([sym, spot, strike,number_of_days,sigma])
		idxThisDayAndTicker = row[0]
		if optPC =="P":
			model_from_price = mibian.BS(bsdata, putPrice=row.PriceBase)
			model_from_hvol = mibian.BS(bsdata, volatility=row.HVOL * 100)
			IVOL =model_from_price.impliedVolatility
			PRICE_HVOL = model_from_hvol.putPrice
			PRICE_IVOL = model_from_price.putPrice
			DELTA_HVOL = model_from_hvol.putDelta
			DELTA_IVOL = model_from_price.putDelta
			TERM_DAYS = number_of_days
			PROB_EXERCISE = 1.0 - mibian.BS(bsdata, volatility=row.HVOL*100).exerciceProbability
			dict_histVol[idxThisDayAndTicker] =  [IVOL, PRICE_HVOL,PRICE_IVOL,DELTA_HVOL , DELTA_IVOL, TERM_DAYS,  PROB_EXERCISE]

		elif optPC =="C":
			# Get Implied Vol from callPrice
			model_from_price = mibian.BS(bsdata, callPrice=row.PriceBase)
			model_from_hvol = mibian.BS(bsdata, volatility=row.HVOL*100)
			IVOL = model_from_price.impliedVolatility
			PRICE_HVOL = model_from_hvol.callPrice
			PRICE_IVOL = model_from_price.callPrice
			DELTA_HVOL = model_from_hvol.callDelta
			DELTA_IVOL = model_from_price.callDelta
			TERM_DAYS = number_of_days
			PROB_EXERCISE = mibian.BS(bsdata, volatility=row.HVOL*100).exerciceProbability

			dict_histVol[idxThisDayAndTicker] = [IVOL, PRICE_HVOL, PRICE_IVOL, DELTA_HVOL, DELTA_IVOL, TERM_DAYS, PROB_EXERCISE]

	computed_option_fields = ["IVOL", "PRICE_HVOL", "PRICE_IVOL", "DELTA_HVOL", "DELTA_IVOL", "TERM_DAYS",
							  "PROB_EXERCISE"]
	dictbs = {idx: model.__dict__ for idx, model in mibianBS.items()}
	#dfbs = pd.DataFrame.from_dict(dictbs, orient='index').reset_index().rename(columns={'level_0':'TradeData', 'level_1':'Ticker'})
	dfoptions = pd.DataFrame.from_dict(dict_histVol,orient='index', columns=computed_option_fields)
	dfoptions.index = pd.MultiIndex.from_tuples(dfoptions.index.values, names=df.index.names)
	return option_models, dfoptions



#These are the trades since 12-31-2020
#The year end positions / weights are in the file

import sqlite3




def load_symbols_from_dash(table="DashSummary_20210920"):
	"""Loads a """
	global data
	conn = sqlite3.connect("bdin.db")
	c = conn.cursor()
	data = pd.read_sql(con=conn, sql=f"SELECT * FROM {table}")
	data = data.rename(columns={'LongName.1':'FirstWordSecurity'})

	return data
def download_returns(df):
	names = list(df['Symbol'].unique())
	rets = {sym: qs.utils.download_returns(sym, period="5y") for sym in list(set(df['Symbol']))}

	return pd.concat(rets,axis=1)

def compute_risk_from_deltas(df):
	"""A function that computes the return for a group"""
	pass


def load_NAV(table):

	conn = sqlite3.connect("bdin.db")
	c = conn.cursor()
	data = pd.read_sql(con=conn, sql=f"SELECT * FROM {table}")

	#returns_NAV = qs.utils.to_log_returns(load_NAV(f'{fund}_NAV')['NAV'])

	return data.query('Class=="F"').copy()


def load_holdings(table):
	"""table: "BDIN_HOLDINGS_20201231"""
	value_date=pd.to_datetime(table.split("_")[2])
	fund = table.split("_")[0]
	global data
	conn = sqlite3.connect("bdin.db")
	c = conn.cursor()
	data = pd.read_sql(con=conn, sql=f"SELECT * FROM {table}")

	#returns_NAV = qs.utils.to_log_returns(load_NAV(f'{fund}_NAV')['NAV'])
	data['Date']=value_date
	return data
def load_market_environments():
	"""Load MarketEnvironment by market"""
	pass

from collections import OrderedDict

def get_equities(df):
	options = OrderedDict()
	for row in df.itertuples():
		assert row.AssetClass == "Equity"
		options[row.Symbol] = EquityModel(
			sym=row.Symbol,
			ISIN=row.ISIN)

	return options


def get_option_objects(df):
	options = {}
	for row in df.itertuples():
		assert row.AssetClass == "Option"
		options[row.ISIN] = OptionModel(value_date=row[0][0],
										sym=row.Symbol,
										sigma=row.HVOL,
										spot=row.Close,
										expiry=row.EXPIRY_DATE,
										strike=row.STRIKE,
										frate=row.RATE,
										qrate=row.RATE_Q,
										optPC=row.OptPC,
										bdfunds_ticker=None)

	return options


def process_trades(df, holdings_start=None):
	"""Clean columns"""
	df.columns = [c.replace("/", "").strip().replace(" ", "").replace("%", "Pct") for c in df.columns]

	dfeq =  df.query("AssetClass==['Equity','Option']").copy()
	dfeq['Symbol'] = dfeq['Ticker'].str.split(" ").map(lambda x: x[0])

	option_tickers_raw = dfeq.query('AssetClass == "Option"')['Ticker'].unique()

	options_split = {ticker: dict(zip(["Symbol","Currency","ExpiryDate","PCStrike"],ticker.split(" "))) for ticker in option_tickers_raw}
	options_split_Expiry = {ticker: pd.to_datetime(opt.get('ExpiryDate')) for ticker,opt in options_split.items()}

	options_split_PutCall = {ticker:opt.get('PCStrike')[0] for ticker,opt in options_split.items()}
	options_split_Strike = {ticker: float(opt.get('PCStrike')[1:]) for ticker, opt in options_split.items()}

	dfeq['OptPC'] = dfeq['Ticker'].map(lambda x:options_split_PutCall.get(x,"C"))

	dfeq['UNDERLYING_LAST_PRICE'] = dfeq['PriceBase']
	dfeq['STRIKE'] = dfeq['Ticker'].map(lambda x: options_split_Strike.get(x, .01)) #StrikePrice or 1 cent for equities
	dfeq['EXPIRY_DATE'] = dfeq['Ticker'].map(lambda x: options_split_Expiry.get(x,pd.to_datetime('2050-01-01'))) #Expiry or some date far in the future for Equity
	dfeq['TERM_DAYS'] = (dfeq['EXPIRY_DATE'] - dfeq['TradeData']).map(lambda x: max(x.days, 0)) #Number of days in Term
	dfeq['TERM_YEARS'] = dfeq['TERM_DAYS']/365.0
	dfeq['RATE'] = .01
	dfeq['RATE_Q'] =.0
	dfeq['X8VOL'] = .25
	dfeq['BSVOL'] = dfeq['X8VOL']*100
	dfeq['Date'] = dfeq['TradeData'].values
	dfeq = dfeq.set_index(['Date','Ticker'])
	return dfeq

def load_trades_from_db(table="trades_BDIN"):
	'''Get trades from db'''
	global data
	conn = sqlite3.connect("bdin.db")
	c = conn.cursor()
	data = pd.read_sql(con=conn, sql=f"SELECT * FROM {table}")
	clean_column_sql = lambda df, col: df[col].map(lambda x: float(str(x).replace(",", "").replace("%","Pct")))
	dftrades = data.assign(
		SharesPar=clean_column_sql(data, 'Shares/Par'),
		PctOfPortfolio =clean_column_sql(data, '%ofPortfolio'),
		FirstWord = data['SecurityDescription'].map(lambda x: x.split(" ")[0]))
	return dftrades





value_date = '2021-09-16'
fp_BDFund = pathlib.Path("Z:\\rshinydata\\summary\\BDFundPortfolio.csv")
def load_BDFunds(fp=fp_BDFund, valdate=value_date):
	"""Loads the BDFunds file from mounted S3"""


	df = pd.read_csv(fp_BDFund, parse_dates=['OptExpiryYYYYMMDD'])
	dfbd = df.assign(
	Weight = df.CurrentWeight/100.0
					 )
	return dfbd


import mibian

import dataclasses

def add_positions_by_date(df):
	"""Accumulates positions in each security"""

	dfpos = pd.pivot_table(df.reset_index(), index='TradeData', columns='SecurityDescription', values='SharesPar', aggfunc="sum", fill_value=0).cumsum()
	return dfpos

def add_risk_metrics(df):
	"""Adds Position Delta , VaR"""


	df['DELTA_UNIT'] = np.where(df['AssetClass'] == "Equity", 1 , df['DELTA_HVOL'])
	df['RISK_SCALING'] = np.where(df['AssetClass'] == "Option", 100 , 1)
	df['RISK_UNITS'] = df['RISK_SCALING'] * df['POS']


	df['POS_DELTA'] = df['RISK_UNITS'] * df['DELTA_UNIT']
	df['DOLLAR_DELTA'] = df['POS_DELTA'] * df['Close']
	return df


def report_from_NAV(fund):
	"""Generates report based on NAV"""
	dfNAV = load_NAV("BDIN_NAV")
	dfNAV[fund]=dfNAV['NAV']
	return dfNAV
import streamlit as st

TARGET_EXPIRY = '2021-12-17'
s3bucket = "s3://blkd/rshinydata/pricedata/"
fps3_BDFund = "s3://blkd/rshinydata/summary/BDFundPortfolio.csv"

path_dash = pathlib.Path( "Z:\\rshinydata\\summary")
path_options = pathlib.Path("Z:\\rshinydata\\currentoptdata")
path_equities = pathlib.Path( "Z:\\rshinydata\\pricedata")
path_options_history = pathlib.Path( "Z:\\rshinydata\\pricedata")
fp_DashSummary = pathlib.Path( "Z:\\rshinydata\\summary\\DashSummary.csv")

fp_TradeList = pathlib.Path( "Z:\\rshinydata\\summary\\SampleTradeList.csv")

options_bad_files = ['CurrentZones - Copy.csv', 'CurrentZones.csv', '_QuoteSummary.csv']

listfiles = lambda path: [p for p in os.listdir(path)]
names_eq = [f.split(".csv")[0] for f in listfiles(path_equities) if not f.endswith('_split.csv')]
tickers = {name: yf.Ticker(name) for name in names_eq}

# <editor-fold desc="Description">
#@pf.register_dataframe_method
#def get_names_eq(df: pd.DataFrame):
#	return list(set(df['Symbol']))
#@pf.register_dataframe_method
#def filter_options(df:pd.DataFrame):
#	return df[df.AssetClass=="Equity"].copy()
#def get_pos_delta(df:pd.DataFrame):
#	return df.query('AssetClass=="Equity"').groupby(['Symbol','Date'])['SharesPar']
# </editor-fold>



if __name__ == "__main__":


	#prices = yf.download(syms,start,end)
	mibianBS={}
	tbl_holdings_start = "BDIN_HOLDINGS_20201231"
	tbl_holdings_end = "BDIN_HOLDINGS_20210913"
	dfpositions_start = load_holdings(tbl_holdings_start)

	dfattribution_ytd = test_load_attribution(filename_attribution_ytd)
	dfattribution_ltd = test_load_attribution(filename_attribution_ltd)
	AUM = pd.read_csv("./BDIN_NAV.csv", parse_dates=['Date']).query("Class=='F'")



	dfpositions_final = load_holdings(tbl_holdings_end)

	dftrades = test_load_trades(filename_trades)
	print("Num Zero Symbols",dftrades['Symbol'].isnull().sum())

	prices = yf.download(list(dftrades['Symbol'].unique()), period="2y")['Close']
	prices = prices.stack()
	prices.name = 'Close'
	prices = prices.reset_index().rename({'level_1': 'Symbol'},axis=1)

	#returns = download_returns()



	dfmerge_price = pd.merge(prices, dftrades.reset_index(), how='right', on=['Date', 'Symbol'])



	returns =download_returns(dfmerge_price)

	vols = get_volatility(returns.fillna(0)).dropna()
	dfvols = pd.DataFrame(vols.stack().reset_index())
	dfvols.columns = ['Date', 'Symbol', 'HVOL']
	dfmerge_vols = pd.merge(dfvols, dfmerge_price, how='right', on=['Date', 'Symbol'])
	dfmerge_vols['HVOL'] = dfmerge_vols['HVOL'].fillna(.55)
	dfmerge_vols = dfmerge_vols.set_index(['TradeData','Ticker'])
	dfmerge_vols_clean = dfmerge_vols[dfmerge_vols.EXPIRY_DATE > dfmerge_vols.Date].copy()


	models, dfoptions = get_options(dfmerge_vols_clean.query('AssetClass=="Option"'))

	dfderiv = dfmerge_vols_clean.join(dfoptions, lsuffix="", rsuffix="_DERIV")
	dfpos = add_positions_by_date(dfderiv).stack()
	dfderiv = dfderiv.reset_index().set_index(['TradeData','SecurityDescription']).join(pd.Series(dfpos,name='POS'))
	dfderiv_risk = add_risk_metrics(dfderiv)




	bsmodel = lambda row: row[['Close', 'Strike', 'RATE', 'TERM_DAYS']]
	def get_days_remaining(model, valdate):
		return (model['ExpiryDate'] - valdate).days




#dfoptions =pd.merge(dfoption_vol.reset_index(),dfoption_prices,on=['TradeData','Ticker'])
#dfoptions = pd.DataFrame.from_dict(orient='index',data={idx:model.__dict__ for idx,model in prices_hvol.items()}).rename(columns={'exerciceProbability':'probExercise'})

# '''Black-Scholes
# Used for pricing European options on stocks without dividends
#
# BS([underlyingPrice, strikePrice, interestRate, daysToExpiration], \
# 		volatility=x, callPrice=y, putPrice=z)
#
# eg:
# 	c = mibian.BS([1.4565, 1.45, 1, 30], volatility=20)
# 	c.callPrice				# Returns the call price
# 	c.putPrice				# Returns the put price
# 	c.callDelta				# Returns the call delta
# 	c.putDelta				# Returns the put delta
# 	c.callDelta2			# Returns the call dual delta
# 	c.putDelta2				# Returns the put dual delta
# 	c.callTheta				# Returns the call theta
# 	c.putTheta				# Returns the put theta
# 	c.callRho				# Returns the call rho
# 	c.putRho				# Returns the put rho
# 	c.vega					# Returns the option vega
# 	c.gamma					# Returns the option gamma
#
# 	c = mibian.BS([1.4565, 1.45, 1, 30], callPrice=0.0359)
# 	c.impliedVolatility		# Returns the implied volatility from the call price
#
# 	c = mibian.BS([1.4565, 1.45, 1, 30], putPrice=0.0306)
# 	c.impliedVolatility		# Returns the implied volatility from the put price
#
# 	c = mibian.BS([1.4565, 1.45, 1, 30], callPrice=0.0359, putPrice=0.0306)
# 	c.putCallParity			# Returns the put-call parity
# 	'''
#
#
# def __init__(self, args, volatility=None, callPrice=None, putPrice=None, \
# 			 performance=None):
# 	self.underlyingPrice = float(args[0])
# 	self.strikePrice = float(args[1])
# 	self.interestRate = float(args[2]) / 100
# 	self.daysToExpiration = float(args[3]) / 365
#
# 	for i in ['callPrice', 'putPrice', 'callDelta', 'putDelta', \
# 			  'callDelta2', 'putDelta2', 'callTheta', 'putTheta', \
# 			  'callRho', 'putRho', 'vega', 'gamma', 'impliedVolatility', \
# 			  'putCallParity']:
# 		self.__dict__[i] = None
#
# 	if volatility:
# 		self.volatility = float(volatility) / 100
#
# 		self._a_ = self.volatility * self.daysToExpiration ** 0.5
# 		self._d1_ = (log(self.underlyingPrice / self.strikePrice) + \
# 					 (self.interestRate + (self.volatility ** 2) / 2) * \
# 					 self.daysToExpiration) / self._a_
# 		self._d2_ = self._d1_ - self._a_
# 		if performance:
# 			[self.callPrice, self.putPrice] = self._price()
# 		else:
# 			[self.callPrice, self.putPrice] = self._price()
# 			[self.callDelta, self.putDelta] = self._delta()
# 			[self.callDelta2, self.putDelta2] = self._delta2()
# 			[self.callTheta, self.putTheta] = self._theta()
# 			[self.callRho, self.putRho] = self._rho()
# 			self.vega = self._vega()
# 			self.gamma = self._gamma()
# 			self.exerciceProbability = norm.cdf(self._d2_)
# 	if callPrice:
# 		self.callPrice = round(float(callPrice), 6)
# 		self.impliedVolatility = impliedVolatility( \
# 			self.__class__.__name__, args, callPrice=self.callPrice)
# 	if putPrice and not callPrice:
# 		self.putPrice = round(float(putPrice), 6)
# 		self.impliedVolatility = impliedVolatility( \
# 			self.__class__.__name__, args, putPrice=self.putPrice)
# 	if callPrice and putPrice:
# 		self.callPrice = float(callPrice)
# 		self.putPrice = float(putPrice)
# 		self.putCallParity = self._parity()

def read_s3_csv(fp, dtype=None, sentinels=None, parse_dates=None):
    #read df from s3
    s3 = S3FileSystem(anon=False)
    #fp = os.path.join('s3://blkd/rshinydata/pricedata/{}'.format(filename))  # df_breederscup.csv'
    df = pd.read_csv(s3.open(fp, mode='rb'),
                     dtype=dtype,
                     na_values=sentinels,
                     parse_dates=parse_dates, keep_date_col=True)
    return df
def load_s3_equity(sym, dtype=None,sentinels=None, parse_dates=None ):
    fp = "s3://blkd/rshinydata/currentoptdata/{}.csv".format(sym)
    s3 = S3FileSystem(anon=False)
    df = pd.read_csv(s3.open(fp, mode='rb'),
                     dtype=dtype,
                     na_values=sentinels,
                     parse_dates=parse_dates, keep_date_col=True)
    return df

def load_s3_options(sym, dtype=None,sentinels=None, parse_dates=None ):
        fp = "s3://blkd/rshinydata/currentoptdata/{}.csv".format(sym)
        s3 = S3FileSystem(anon=False)
        df = pd.read_csv(s3.open(fp, mode='rb'),
                         dtype=dtype,
                         na_values=sentinels,
                         parse_dates=parse_dates, keep_date_col=True)
        return df


def get_option_data(df):
    option_data = {}
    for sym in df.Symbol.unique():
        try:
            option_data[sym] = load_s3_options(sym)
        except:
            pass
    return pd.concat(option_data, axis=0)
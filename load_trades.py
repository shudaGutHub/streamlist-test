import pandas as pd
import numpy as np
import pathlib
import scipy.stats as stats
import pathlib as pathlib
from mibian import BS
import yfinance as yf
from BDFunds import *
DIR_DATA = pathlib.Path('C:\\Users\\salee\\projects\\streamlit-example')

filename_holdings_start="BDIN_HOLDINGS_2020.xlsx"
TABLE_HOLDINGS_END = 'BDIN_HOLDINGS_20210913'
filename_tickers="tickers.csv"

filename_attribution_ytd = "BDIN_Security_YTD.xlsx"
filename_attribution_ltd = "BDIN_Security_LTMONTH.xlsx"
filename_holdings_end="BDIN__HOLDINGS_2021_Aug.xlsx"
filename_trades = 'BDIN_Trades_20210916.xlsx'
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
		self.underlying = yf.Ticker(self.sym)

	# self.underlying = BDEquity(sym, start)
	def load_s3(self):
		pass
		#self.data_s3 = load_s3_options(self.sym).query(
		#'ExpirationYYYYMMDD==@self.expiry & strike==@self.strike & pc==@self.OptPC')

	def term_in_years(self):
		return (self.expiry - self.value_date).days / 365.0

	def get_yahoo(self):
		test_expiry = self.EXPIRY_YF.strftime("%Y-%m-%d")
		# test_expiry not in self.underlying_ticker.option_chain(test_expiry).calls.query("strike==@self.strike") #.strftime("%Y-%m-%d")- (pd.Timedelta(1, 'DAYS'))

		try:
			if self.optPC == "C":
				return self.underlying_ticker.option_chain(test_expiry).calls.query("strike==@self.strike")
			if self.optPC == "P":
				return self.underlying_ticker.option_chain(test_expiry).puts.query("strike==@self.strike")
			else:
				print("Not P or C")
				return None
		except:
			test_expiry = self.expiry.date().strftime("%Y-%m-%d")
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
	prices = yf.download(symbols_options, start_date, end_date)
	return prices

def get_returns_from_prices(prices):
	return qs.utils.to_log_returns(prices)


def get_volatilities_from_returns(returns):
	return qs.stats.volatility(returns)

def test_load_trades(filename):
	df = pd.read_excel(
	io = get_fp(filename),
	engine = 'openpyxl',
	sheet_name = 'BDIN_TRADES',
	skiprows = 0,  # TODO Process from raw file requires stripping top row
	usecols = 'A:M',
	parse_dates = ['Trade Data', 'Effective Date'],

	)
	df.columns = [c.replace("/", "").strip().replace(" ", "").replace("%", "Pct") for c in df.columns]

	dfeq =  df.query("AssetClass==['Equity','Option']").copy()
	dfeq['Symbol'] = dfeq['Ticker'].str.split(" ").map(lambda x: x[0])

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
def test_load_NAV(filename):
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


		bsdata = [row.Close,
				  row.STRIKE, #This is a fixed number for the life of the contract
				  row.RATE * 100,
				  number_of_days]

		print([sym, spot, strike,number_of_days,sigma])
		idxThisDayAndTicker = row[0]
		if optPC =="P":
			#Get Implied Vol from putPrice

			dict_impliedVol[idxThisDayAndTicker, f'impliedVolatility_{optPC}'] = mibian.BS(bsdata, putPrice=row.PriceBase).impliedVolatility
			mibianBS[idxThisDayAndTicker] = mibian.BS(bsdata,volatility=sigma*100)

		elif optPC =="C":
			# Get Implied Vol from callPrice
			dict_impliedVol[idxThisDayAndTicker, f'impliedVolatility_{optPC}'] = mibian.BS(bsdata, callPrice=row.PriceBase).impliedVolatility
			mibianBS[idxThisDayAndTicker] = mibian.BS(bsdata, volatility = row.HVOL * 100)

		else:
			mibianBS[idxThisDayAndTicker] = None


	dictbs = {idx: model.__dict__ for idx, model in mibianBS.items()}
	dfbs = pd.DataFrame.from_dict(dictbs, orient='index').reset_index().rename(columns={'level_0':'TradeData', 'level_1':'Ticker'})

	return dict_impliedVol,dfbs



#These are the trades since 12-31-2020
#The year end positions / weights are in the file

import sqlite3




def load_holdings(table):

	global data
	conn = sqlite3.connect("bdin.db")
	c = conn.cursor()
	data = pd.read_sql(con=conn, sql=f"SELECT * FROM {table}")
	clean_column_sql = lambda df, col: df[col].map(lambda x: float(str(x).replace(",", "")))
	dfpositions = data.assign(
		SharesPar=clean_column_sql(data, 'Shares/Par')
	)
	return dfpositions





import mibian

import dataclasses


def add_risk_metrics(df):
	"""Adds Position Delta , VaR"""

	df['DELTA_UNIT'] = np.where(df['OptPC']=="P", df['putDelta'], df['callDelta'])
	df['DELTA_UNIT'] = np.where(df['AssetClass'] == "Equity",1 , df['DELTA_UNIT'])
	df['RISK_SCALING'] = np.where(df['AssetClass'] == "Option",100 , 1)
	df['RISK_UNITS'] = df['RISK_SCALING']*df['SharesPar']
	df['POS_DELTA'] = df['RISK_UNITS']*df['DELTA_UNIT']
	df['DOLLAR_DELTA'] = df['POS_DELTA'] * df['underlyingPrice']
	return df


if __name__ == "__main__":
	impliedVols = {}

	#prices = yf.download(syms,start,end)
	mibianBS={}
	tbl_holdings_start = "BDIN_HOLDINGS_20201231"
	tbl_holdings_end = "BDIN_HOLDINGS_20210913"
	dfpositions_start = load_holdings(tbl_holdings_start)


	dfpositions_final = load_holdings(tbl_holdings_end)

	dftrades = test_load_trades(filename_trades)


	prices = load_prices_underlyings(dftrades)
	dfprices = prices['Close'].stack().reset_index().drop_duplicates()
	dfprices.columns = ['Date', 'Symbol', 'Close']
	dfmerge_price = pd.merge(dfprices, dftrades.reset_index(), how='right', on=['Date', 'Symbol'])

	dfattribution_ytd = test_load_attribution(filename_attribution_ytd)
	dfattribution_ltd = test_load_attribution(filename_attribution_ltd)
	dfnav = test_load_NAV(filename_NAV)

	returns = get_returns(prices['Adj Close'])
	vols = get_volatility(returns)
	dfvols = vols.stack().reset_index()

	dfvols.columns = ['Date', 'Symbol', 'HVOL']
	dfmerge_vols = pd.merge(dfvols, dfmerge_price, how='right', on=['Date', 'Symbol'])
	dfmerge_vols['HVOL'] = dfmerge_vols['HVOL'].fillna(.5)
	dfmerge_vols = dfmerge_vols.set_index(['TradeData','Ticker'])
	dfmerge_vols_clean = dfmerge_vols[dfmerge_vols.EXPIRY_DATE > dfmerge_vols.Date].copy()
	dfoptions, dfmodel = get_options(dfmerge_vols_clean.query('AssetClass=="Option"'))

	dfmodel = dfmodel.rename(columns={'exerciceProbability':'probCall'})


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
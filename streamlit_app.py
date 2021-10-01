import os
from collections import namedtuple
import altair as alt
import math
import pathlib
import pandas as pd
import sqlalchemy
import streamlit as st

import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import sqlite3

import pandas as pd

import mibian

from pandas import concat, DataFrame, read_csv, to_datetime, Series, MultiIndex
import sqlite3

import load_trades
fp_DashSummary = pathlib.Path( "Z:\\rshinydata\\summary\\DashSummary.csv")
fp_BDFund = pathlib.Path("Z:\\rshinydata\\summary\\BDFundPortfolio.csv")
fp_TradeList = pathlib.Path("Z:\\rshinydata\\summary\\SampleTradeList.csv")
path_dash = pathlib.Path("Z:\\rshinydata\\summary")
path_options = pathlib.Path("Z:\\rshinydata\\currentoptdata")
path_equities = pathlib.Path("Z:\\rshinydata\\pricedata")


conn = sqlite3.connect("bdin.db")
value_date = st.date_input('Value Date')

dfDashSummary = load_trades.load_symbols_from_dash().dropna(subset=['LongName'])
fp_BDFund = pathlib.Path("Z:\\rshinydata\\summary\\BDFundPortfolio.csv")
def load_BDFunds(fp=fp_BDFund, valdate=value_date):
	"""Loads the BDFunds file from mounted S3"""

	df = pd.read_csv(fp_BDFund, parse_dates=['OptExpiryYYYYMMDD'])
	dfbd = df.assign(
	Weight = df.CurrentWeight/100.0,
	Symbol = df['Ticker'].map(lambda x: x.split(" ")[0]),
	CountryCode = df['Ticker'].map(lambda x: x.split(" ")[1])
			)

	return dfbd.query('AssetType==["EQ","OP"]')
dfBDFunds =load_BDFunds(fp_BDFund)


sql_options = 'SELECT * FROM BDFunds_20210930 WHERE AssetType = "OP" AND Portfolio="BDOP_Portfolio"'
pos_deriv = pd.read_sql(con=conn,sql=sql_options)

symbols = dfDashSummary['Symbol']

def get_options(df,
                attr_price='PriceBase', attr_vol="HVOL",  start='2017-01-01', end='2021-09-14'):
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




#S([underlyingPrice, strikePrice, interestRate, daysToExpiration],
def load_option_prices_from_S3(pathS3=path_options):
    """Load pathS3 options """
    # option_data = {f.replace(".csv",""): pd.read_csv(f) for f in os.listdir(path_options) if f not in options_bad_files}
    options_bad_files = ['CurrentZones - Copy.csv', 'CurrentZones.csv', '_QuoteSummary.csv']
    DTYPES_OPTIONS3 = {'Symbol': 'str',
                       'idx': 'int',
                       'contractSymbol': 'str',
                       'strike': 'float',
                       'currency': 'str',
                       'lastPrice': 'float',
                       'change': 'float',
                       'percentChange': 'float',
                       'volume': 'float',
                       'openInterest': 'float',
                       'bid': 'float',
                       'ask': 'float',
                       'contractSize': 'float',
                       'expiration': 'str',
                       'lastTradeDate': 'str',
                       'impliedVolatility': 'float',
                       'inTheMoney': 'str',
                       'symbol': 'str',
                       'pc': 'str',
                       'RegularMarketPrice': 'float',
                       'StrikeRatio': 'float',
                       'ExpirationYYYYMMDD': 'str',
                       'LastTradeYYYYMMDD': 'str',
                       'MarketPriceYYYYMMDD': 'str',
                       'NumDaysToExpiry': 'float',
                       'NumDaysSinceTrade': 'float',
                       'ImpliedYield': 'float',
                       'ImpliedYield4': 'float'}

    def load_option_file(f, dtypes):

        """Load an options file from s3 """
        data = pd.read_csv(pathlib.Path(f))
                           #parse_dates=['ExpirationYYYYMMDD', 'LastTradeYYYYMMDD', 'expiration', 'lastTradeDate'])

        return data
    files = [f for f in os.listdir(pathS3)  if f not in options_bad_files]
    option_data = {}
    for f in files:
        sym = f.replace(".csv", "")
        option_data[sym] = load_option_file(pathlib.Path(pathS3,f), dtypes=DTYPES_OPTIONS3)

    # if len(data) < 1
    #     print(data.head(3))
    #     df = data.assign(
    #         OptPC=data.pc.str.upper(),
    #         EXPIRY=data.ExpirationYYYYMMDD,
    #         TERM_DAYS=data.NumDaysToExpiry,
    #         TERM_YEARS=data.NumDaysToExpiry / 365.0,
    #         rd=0.0,
    #         rq=0.0
    #     )
    return pd.concat(option_data,names=['Symbol','idx'])

options = load_option_prices_from_S3()


def create_dashsummary_table(df=dfDashSummary):
    # Import the bar csv file into a dataframe
    df['Date'] = pd.to_datetime('now')
    df = df.fillna(0)
    df['updated'] = pd.to_datetime('now')

    # Write the data into the database, this is so fucking cool
    df.to_sql('dashsummary', engine, if_exists='replace', index=False)


    return 'Daily prices table created'


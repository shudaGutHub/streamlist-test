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


dfBDFunds = load_trades.load_BDFunds(load_trades.fp_BDFund,value_date)
dfDashSummary = load_trades.load_symbols_from_dash().dropna(subset=['LongName'])


symbols = dfDashSummary['Symbol']

tables = {'DashSummary': dfDashSummary,'BDFundPortfolio':dfBDFunds}


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


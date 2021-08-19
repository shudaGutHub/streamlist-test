from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st

import os
from pandas import concat, DataFrame, read_csv, to_datetime, Series, MultiIndex
# from datetime import timedelta, datetime

from fractions import Fraction
import matplotlib.pyplot as plt
import seaborn as sns
from s3fs.core import S3FileSystem
import xlwings as xw
import datetime as dt
import yfinance as yf
import openpyxl
import scipy.stats as stats
from collections import namedtuple

from openpyxl import workbook
import quantstats as qs
# %%
import sys
sys.path.insert(1,"C:\\Users\\Saleem\\projects\\X8MKB\\x8mkb")
from BDFunds import get_option_data, load_s3_options

# %%

value_date = '2021-08-17'
target_expiry = '2021-10-17'

# %%

# This is the bucket for price data
# s3bucket = "s3://blkd/rshinydata/pricedata/"
fps3_BDFund = "s3://blkd/rshinydata/summary/BDFundPortfolio.csv"
fps3_Dash = "s3://blkd/rshinydata/summary/DashSummary.csv"
# "s3://blkd/rshinydata/summary/{}".format(bdfund_filename)
# BDFundPortfolio_filename = "https://blkd.s3.us-east-2.amazonaws.com/rshinydata/summary/BDFundPortfolio.csv"
# DashSummary_filename = "https://blkd.s3.us-east-2.amazonaws.com/rshinydata/summary/DashSummary.csv"

s3summary = "s3://blkd/rshinydata/summary/"
fps3_BDCCSummary = os.path.join(s3summary, "BDCCSummary.csv")
fps3_CPSummary = os.path.join(s3summary, "CPSummary.csv")


# dfDash = read_s3_csv(fps3_Dash)
# dfBDCC = read_s3_csv(fps3_BDCCSummary)
# dfCP = read_s3_csv(fps3_CPSummary)
# dfDash = read_s3_csv(DashSummary_filename)

# %%

def load_s3_equity(sym, dtype=None, sentinels=None, parse_dates=None):
    fp = "s3://blkd/rshinydata/pricedata/{}.csv".format(sym)
    s3 = S3FileSystem(anon=False)
    try:
        df = pd.read_csv(s3.open(fp, mode='rb'),
                         dtype=dtype,
                         na_values=sentinels,
                         parse_dates=parse_dates, keep_date_col=True)
    except:
        print("No data: {}", sym)
        return None
    return df




# %% md

# Get names for BDEQ


# %%

import csv


def read_s3_csv(fp, dtype=None, sentinels=None, parse_dates=None, error_bad_lines=False):
    # read df from s3
    s3 = S3FileSystem(anon=False)
    # fp = os.path.join('s3://blkd/rshinydata/pricedata/{}'.format(filename))  # df_breederscup.csv'
    df = pd.read_csv(s3.open(fp, mode='rb'),
                     engine='python',
                     dtype=dtype,
                     na_values=sentinels,
                     parse_dates=parse_dates, keep_date_col=True, error_bad_lines=error_bad_lines
                     )
    return df


# %%


# %%

dfDashS3 = read_s3_csv(fps3_Dash, dtype={'OptionExpiry': str, 'LastStockPriceDate': str, 'ZoneYYYYMMDD': str},
                       sentinels={'OptionExpiry': ['0', 'NA', '']},
                       parse_dates=['OptionExpiry', 'LastStockPriceDate', 'ZoneYYYYMMDD'], error_bad_lines=False)

dfBDFunds = read_s3_csv(fps3_BDFund)   #"s3://blkd/rshinydata/summary/BDFundPortfolio.csv"
dtype_bd = {'OptExpiryYYYYMMDD': str, 'Units': float}
sentinels_bd = {'OptExpiryYYYYMMDD': ['0', 'NA', '']}
parse_dates_bd = ['OptExpiryYYYYMMDD']

# Magic commands implicitly `st.write()`
''' _This_ is some __Markdown__
BlackDiamond Risk Dashboard'''


import pathlib
import numpy as np

value_date = st.date_input('Value Date')
pathTQ= os.path.join("C:\\Users\\Saleem\\OneDrive\\Documents\\blkd\\{}".format(value_date))

fund = st.selectbox('Select Fund', ['BDEQ_Portfolio','BDOP_Portfolio','BDIN_Portfolio'])
deriv = st.radio('Include Deriv',['Yes','No'])
data = pd.read_csv(pathlib.Path(pathTQ,fund+".csv"))
st.dataframe(data)

#prices = yf.download(tickers=symbol,period='1y')
#st.line_chart(prices)
#positions = data.query('Symbol = @symbol')['Units']
#st.bar_chart(data=positions)



    
    
    
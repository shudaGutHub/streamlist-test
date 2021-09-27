from collections import namedtuple
import altair as alt
import math
import pathlib
import pandas as pd
import streamlit as st
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import sqlite3

import pandas as pd


from pandas import concat, DataFrame, read_csv, to_datetime, Series, MultiIndex
import sqlite3

conn = sqlite3.connect("bdin.db")
value_date = st.date_input('Value Date')

#URL_DASH = "https://blkd.s3.us-east-2.amazonaws.com/rshinydata/summary/DashSummary.csv"
fund = st.selectbox('Select Fund', ['BDEQ_Portfolio','BDOP_Portfolio','BDIN_Portfolio','BDOP_DERIV'])

data = pd.read_sql(con=conn, sql="SELECT * FROM  main_BDFundPortfolio")

#symbols = list(data['Symbol'].unique())

#dfequity = data[data.AssetType=="EQ"].copy()
#risk = st.sidebar.multiselect("Risk Factor", symbols )

st.dataframe(data.query("Portfolio == @fund"))



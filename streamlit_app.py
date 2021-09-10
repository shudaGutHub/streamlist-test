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

import pandas as pd

# from datetime import timedelta, datetime




# Magic commands implicitly `st.write()`
''' BlackDiamond Risk Dashboard'''


value_date = st.date_input('Value Date')

#URL_DASH = "https://blkd.s3.us-east-2.amazonaws.com/rshinydata/summary/DashSummary.csv"

option = st.sidebar.selectbox('Select one symbol', ( 'AAPL', 'MSFT',"SPY",'WMT'))
fund = st.selectbox('Select Fund', ['BDEQ_Portfolio','BDOP_Portfolio','BDIN_Portfolio','BDOP_Holdings_TQ.csv'])
file = st.file_uploader("Choose File")

data = pd.read_csv(f"{file}")
#dfoptions = data[data.AssetType=="OP"].copy()
#dfequity = data[data.AssetType=="EQ"].copy()

st.dataframe(data)





#prices = yf.download(tickers=symbol,period='1y')
#st.line_chart(prices)
#positions = data.query('Symbol = @symbol')['Units']
#st.bar_chart(data=positions)



    
    
    

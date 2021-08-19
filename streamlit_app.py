from collections import namedtuple
import altair as alt
import math
import pathlib
import pandas as pd
import streamlit as st

from pandas import concat, DataFrame, read_csv, to_datetime, Series, MultiIndex
# from datetime import timedelta, datetime




# Magic commands implicitly `st.write()`
''' _This_ is some __Markdown__
BlackDiamond Risk Dashboard'''


value_date = st.date_input('Value Date')
#pathTQ= pathlib.Path("C:\\Users\\Saleem\\OneDrive\\Documents\\blkd\\{}".format(value_date))
#URL_DASH = "https://blkd.s3.us-east-2.amazonaws.com/rshinydata/summary/DashSummary.csv"
fund = st.selectbox('Select Fund', ['BDEQ_Portfolio','BDOP_Portfolio','BDIN_Portfolio'])
deriv = st.radio('Include Deriv',['Yes','No'])
data = pd.read_csv(f"{fund}.csv")
st.dataframe(data.set_index(['Symbol','WEIGHT_USD']))

#prices = yf.download(tickers=symbol,period='1y')
#st.line_chart(prices)
#positions = data.query('Symbol = @symbol')['Units']
#st.bar_chart(data=positions)



    
    
    

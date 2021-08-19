from collections import namedtuple
import altair as alt
import math
import pathlib
import pandas as pd
import streamlit as st
import os
from pandas import concat, DataFrame, read_csv, to_datetime, Series, MultiIndex
# from datetime import timedelta, datetime




# Magic commands implicitly `st.write()`
''' _This_ is some __Markdown__
BlackDiamond Risk Dashboard'''


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



    
    
    

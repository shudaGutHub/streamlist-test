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


value_date = st.date_input('Value Date')

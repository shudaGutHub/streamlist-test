import pandas as pd  #pip install pandas openpyxl
import streamlit as st


dftrades = pd.read_excel(
	io='BDIN_BUYS.xlsx',
	engine='openpyxl',
	sheet_name='BDIN_BUYS',
	skiprows=0,
	usecols='A:N',
	nrows=390,
)
dfholdings = pd.read_excel(
	io='BDIN_HOLDINGS.xlsx',
	engine='openpyxl',
	sheet_name='BDIN_BUYS',
	skiprows=0,
	usecols='A:N',
	nrows=390,
)



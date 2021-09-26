# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import sqlite3
from _csv import reader
import openpyxl
import streamlit as st
import pandas as pd
import pathlib
from helpers import convert_to_USD, convert_dates_YYYY_mm_dd, checkmerge
import helpers

fp_DashSummary = pathlib.Path("Z:\\rshinydata\\summary\\DashSummary.csv")
fp_BDFund = pathlib.Path("Z:\\rshinydata\\summary\\BDFundPortfolio.csv")
fp_TradeList = pathlib.Path("Z:\\rshinydata\\summary\\SampleTradeList.csv")

path_dash = pathlib.Path("Z:\\rshinydata\\summary")
path_options = pathlib.Path("Z:\\rshinydata\\currentoptdata")
path_equities = pathlib.Path("Z:\\rshinydata\\pricedata")
path_options_history = pathlib.Path("Z:\\rshinydata\\pricedata")

path_dash = pathlib.Path("Z:\\rshinydata\\summary")
path_options = pathlib.Path("Z:\\rshinydata\\currentoptdata")
path_equities = pathlib.Path("Z:\\rshinydata\\pricedata")
path_options_history = pathlib.Path("Z:\\rshinydata\\pricedata")

options_bad_files = ['CurrentZones - Copy.csv', 'CurrentZones.csv', '_QuoteSummary.csv']

pathTQ = pathlib.Path('C:\\Users\\salee\\projects\\streamlit-example')

TABLE_HOLDINGS_END = 'BDIN_HOLDINGS_20210913'

path_db = pathlib.Path("C:/Users/salee/projects/streamlit-example/bdin.db")
def load_holdings_TQ(fund, pathTQ=pathTQ):
    return pd.read_excel(
        io=pathlib.Path(pathTQ, f"{fund}_Holdings.xlsx"),
        engine='openpyxl',
        sheet_name='Sheet1',
        skiprows=0,
        usecols='A:W')

from helpers import force_float

def load_holdings_db(fund, path_db=path_db, dtypes=None):
    return

def load_holdings(table):
    """table: "BDIN_HOLDINGS_20201231"""
    value_date = pd.to_datetime(table.split("_")[2])
    fund = table.split("_")[0]
    global data
    conn = sqlite3.connect(path_db)
    c = conn.cursor()
    data = pd.read_sql(con=conn, sql=f"SELECT * FROM {table}")
    # returns_NAV = qs.utils.to_log_returns(load_NAV(f'{fund}_NAV')['NAV'])
    data['Date'] = value_date
    return data

def load_trades_db(fund, path_db=path_db, ):
    """Loads TrueQuant exports from database
    fund: "BDIN"
    conn: conn = sqlite3.connect("bdin.db")
    trades_BDIN
    trades_BDEQ
    trades_BDOP"""
    global data

    conn = sqlite3.connect(path_db)
    table = f"trades_{fund}"
    data = pd.read_sql(con=conn, sql=f"SELECT * FROM {table}")
    data['TradeDate'] = helpers.convert_dates_YYYY_mm_dd(data,'TradeData')
    data['EffectiveDate'] = helpers.convert_dates_YYYY_mm_dd(data, 'TradeData')
    data['Date'] = helpers.convert_dates_YYYY_mm_dd(data, 'TradeData')
    data['TradeValue'] = helpers.force_float(data, 'Value')
    # returns_NAV = qs.utils.to_log_returns(load_NAV(f'{fund}_NAV')['NAV'])
    return data

def load_risk_factors(dftrades):
    """Finds the risk factors"""
    dftrades

def load_holdings(table):
    """table: "BDIN_HOLDINGS_20201231"""
    value_date = pd.to_datetime(table.split("_")[2])
    fund = table.split("_")[0]
    global data
    conn = sqlite3.connect("bdin.db")
    c = conn.cursor()
    data = pd.read_sql(con=conn, sql=f"SELECT * FROM {table}")

    # returns_NAV = qs.utils.to_log_returns(load_NAV(f'{fund}_NAV')['NAV'])
    data['Date'] = value_date
    return data


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


def load_funds_from_S3():
    funds = ["BDEQ", "BDIN", "BDOP"]
    funds_holdings = {f: load_holdings_TQ(fund=f) for f in funds}
    return funds_holdings

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Load positions
    print("Loading history for BDIN")
    # data_DashSummary_raw = load_csv(fp_DashSummary)
    data_BDFund_raw = load_csv(fp_BDFund)
    data_TradeList_raw = load_csv(fp_TradeList)
    funds = ["BDEQ", "BDIN", "BDOP"]
    funds_holdings = load_funds_from_S3()
    holdings_BDIN = funds_holdings.get('BDIN')
    trades_BDIN = load_trades_db('BDIN')



# merge location and country data
    checkmerge(trades_BDIN,holdings_BDIN , mergebyleft="Ticker", mergebyright="Symbol")

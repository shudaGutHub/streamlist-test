import sqlite3
import pandas as pd
import helpers
import pathlib
import numpy as np




def add_risk_multiplier(df):
    df['RISK_MULTIPLIER'] = np.where(df["AssetClass"]=="Option",100,1)
    df['RISK_UNITS'] =df['RISK_MULTIPLIER'] * df['SharesPar']
    return df

def load_trades_db(table, path_db):
    """Loads TrueQuant exports from database
    fund: "BDIN"
    conn: conn = sqlite3.connect("bdin.db")
    trades_BDIN
    trades_BDEQ
    trades_BDOP"""


    conn = sqlite3.connect(path_db)
    table = table
    
    data = pd.read_sql(con=conn, sql=f"SELECT * FROM {table}", parse_dates=['Trade Data','Effective Date'])
    data.columns = [c.replace("/", "").strip().replace(" ", "").replace("%", "Pct") for c in data.columns]
    data['TradeDate'] = helpers.convert_dates_YYYY_mm_dd(data, 'TradeData')
    data['EffectiveDate'] = helpers.convert_dates_YYYY_mm_dd(data, 'TradeData')
    data['Date'] = helpers.convert_dates_YYYY_mm_dd(data, 'TradeData')
    data['ValueTrade'] = helpers.force_float(data, 'Value')

    data= add_risk_multiplier(data)

    # returns_NAV = qs.utils.to_log_returns(load_NAV(f'{fund}_NAV')['NAV'])
    return data






if __name__=="__main__":
    trades = load_trades_db('trades_BDEQ')
    trades_equity = trades.groupby("AssetClass").get_group("Equity").copy()
    trades_option = trades.groupby("AssetClass").get_group("Option").copy()
    print (trades_equity.head())
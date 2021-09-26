import pandas as pd
import sqlite3
from yahoofinancials import YahooFinancials
force_float = lambda df,col: pd.to_numeric(df[col], errors='coerce')
convert_dates_YYYY_mm_dd= lambda df,col: pd.to_datetime(df[col], format='%Y-%m-%d')
convert_to_USD = lambda df, colValue,colFX: df[colValue]*1/df[colFX]

currencies = ['EURUSD=X', 'JPY=X', 'GBPUSD=X']
yahoo_financials_currencies =YahooFinancials(currencies)
def load_symbols_from_dash(table="DashSummary_20210920"):
    """Loads a """
    global data
    conn = sqlite3.connect("C:/Users/salee/projects/streamlit-example/bdin.db")
    c = conn.cursor()
    data = pd.read_sql(con=conn, sql=f"SELECT * FROM {table}")
    data = data.rename(columns={'LongName.1': 'FirstWordSecurity'})
    return data


# check the merge-by column matches
def checkmerge(dfleft, dfright, mergebyleft, mergebyright):
  dfleft['inleft'] = "Y"
  dfright['inright'] = "Y"
  dfboth = pd.merge(dfleft[[mergebyleft,'inleft']],\
    dfright[[mergebyright,'inright']], left_on=[mergebyleft],\
    right_on=[mergebyright], how="outer")
  dfboth.fillna('N', inplace=True)
  print(pd.crosstab(dfboth.inleft, dfboth.inright))
  print(dfboth.loc[(dfboth.inleft=='N') | (dfboth.inright=='N')].head(20))


dfDashSummary = load_symbols_from_dash()
FXmap = dfDashSummary.set_index('Currency')['ExchangeRate'].drop_duplicates()

if __name__ == '__main__':
    pass

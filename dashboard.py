import streamlit as st
import pandas as pd
import numpy as np
import requests
import tweepy
import config 
import psycopg2, psycopg2.extras
import plotly.graph_objects as go
import sqlite3
import pandas as pd
import helpers
import pathlib
import numpy as np
import streamlit as st
from helpers import Instrument as FI


def add_risk_multiplier(df):
    df['RISK_MULTIPLIER'] = np.where(df["AssetClass"] == "Option", 100, 1)
    df['RISK_UNITS'] = df['RISK_MULTIPLIER'] * df['SharesPar']
    return df
auth = tweepy.OAuthHandler(config.TWITTER_CONSUMER_KEY, config.TWITTER_CONSUMER_SECRET)
auth.set_access_token(config.TWITTER_ACCESS_TOKEN, config.TWITTER_ACCESS_TOKEN_SECRET)
api = tweepy.API(auth)
import sqlite3
connection = sqlite3.connect("#DataSourceSettings#
#LocalDataSource: bdin.db
#BEGIN#
<data-source source="LOCAL" name="bdin.db" uuid="eb7eed81-eeac-45d0-b491-49b219a6a5db"><database-info product="SQLite" version="3.34.0" jdbc-version="2.1" driver-name="SQLite JDBC" driver-version="3.34.0" dbms="SQLITE" exact-version="3.34.0" exact-driver-version="3.34"><identifier-quote-string>&quot;</identifier-quote-string></database-info><case-sensitivity plain-identifiers="mixed" quoted-identifiers="mixed"/><driver-ref>sqlite.xerial</driver-ref><synchronize>true</synchronize><jdbc-driver>org.sqlite.JDBC</jdbc-driver><jdbc-url>jdbc:sqlite:$USER_HOME$/OneDrive/bdin.db</jdbc-url><secret-storage>master_key</secret-storage><auth-provider>no-auth</auth-provider><schema-mapping><introspection-scope><node kind="schema" qname="@"/></introspection-scope></schema-mapping><working-dir>$ProjectFileDir$</working-dir></data-source>
#END#

.db")

    #host=config.DB_HOST, database=config.DB_NAME, user=config.DB_USER, password=config.DB_PASS)
#cursor = connection.cursor(cursor_factory=psycopg2.extras.DictCursor)

def load_BDFundPortfolio():
	bdfp = pd.read_sql(con=connection,sql="SELECT * FROM BDFundPortfolio WHERE AssetType = EQ  ORDER BY CurrentWeight DESC")
	return bdfp

	
def add_risk_multiplier(df):
    df['RISK_MULTIPLIER'] = np.where(df["AssetClass"] == "Option", 100, 1)
    df['RISK_UNITS'] = df['RISK_MULTIPLIER'] * df['SharesPar']
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

    data = pd.read_sql(con=conn, sql=f"SELECT * FROM {table}")
    data.columns = [c.replace("/", "").strip().replace(" ", "").replace("%", "Pct") for c in data.columns]
    data['TradeData'] = helpers.convert_dates_YYYY_mm_dd(data, 'TradeData')
    data['EffectiveDate'] = helpers.convert_dates_YYYY_mm_dd(data, 'EffectiveDate')
    data['Date'] = helpers.convert_dates_YYYY_mm_dd(data, 'TradeData')
    data['ValueTrade'] = helpers.force_float(data, 'Value')
    data['SharesPar'] = helpers.force_float(data, 'SharesPar')
    data['PriceBase'] = helpers.force_float(data, 'PriceBase')
    data['PriceLocal'] = helpers.force_float(data, 'PriceLocal')
    data['FirstTradeDate'] = data.groupby('Ticker')['TradeData'].transform(lambda d:d.min())

    data = add_risk_multiplier(data)
    df = process_trades(data)
    return df


def process_trades(df, holdings_start=None):
    """Clean columns"""
    df.columns = [c.replace("/", "").strip().replace(" ", "").replace("%", "Pct") for c in df.columns]

    dfeq = df.query("AssetClass==['Equity','Option']").copy()
    dfeq['Symbol'] = dfeq['Ticker'].str.split(" ").map(lambda x: x[0])

    option_tickers_raw = dfeq.query('AssetClass == "Option"')['Ticker'].unique()

    options_split = {ticker: dict(zip(["Symbol", "Currency", "ExpiryDate", "PCStrike"], ticker.split(" "))) for ticker
                     in option_tickers_raw}
    options_split_Expiry = {ticker: pd.to_datetime(opt.get('ExpiryDate')) for ticker, opt in options_split.items()}

    options_split_PutCall = {ticker: opt.get('PCStrike')[0] for ticker, opt in options_split.items()}
    options_split_Strike = {ticker: float(opt.get('PCStrike')[1:]) for ticker, opt in options_split.items()}

    dfeq['OptPC'] = dfeq['Ticker'].map(lambda x: options_split_PutCall.get(x, "C"))

    dfeq['UNDERLYING_LAST_PRICE'] = dfeq['PriceBase']
    dfeq['STRIKE'] = dfeq['Ticker'].map(
        lambda x: options_split_Strike.get(x, .01))  # StrikePrice or 1 cent for equities
    dfeq['EXPIRY_DATE'] = dfeq['Ticker'].map(lambda x: options_split_Expiry.get(x, pd.to_datetime(
        '2050-01-01')))  # Expiry or some date far in the future for Equity
    dfeq['TERM_DAYS'] = (dfeq['EXPIRY_DATE'] - dfeq['TradeData']).map(
        lambda x: max(x.days, 0))  # Number of days in Term
    dfeq['TERM_YEARS'] = dfeq['TERM_DAYS'] / 365.0
    dfeq['RATE'] = .01
    dfeq['RATE_Q'] = .0
    dfeq['X8VOL'] = .25
    dfeq['BSVOL'] = dfeq['X8VOL'] * 100
    dfeq['Date'] = dfeq['TradeData'].values

    dfeq = dfeq.set_index(['Date', 'Ticker'])
    return dfeq


def get_historical_weights():
    """"""
    pass



#option = st.sidebar.selectbox("Which Dashboard?", ('twitter', 'wallstreetbets', 'stocktwits', 'chart', 'pattern'), 3)

st.header(option)



# if option == 'twitter':
#     for username in config.TWITTER_USERNAMES:
#         user = api.get_user(username)
#         tweets = api.user_timeline(username)
#
#         st.subheader(username)
#         st.image(user.profile_image_url)
#
#         for tweet in tweets:
#             if '$' in tweet.text:
#                 words = tweet.text.split(' ')
#                 for word in words:
#                     if word.startswith('$') and word[1:].isalpha():
#                         symbol = word[1:]
#                         st.write(symbol)
#                         st.write(tweet.text)
#                         st.image(f"https://finviz.com/chart.ashx?t={symbol}")
#
# if option == 'chart':
#     symbol = st.sidebar.text_input("Symbol", value='MSFT', max_chars=None, key=None, type='default')
#
#     data = pd.read_sql("""
#         select date(day) as day, open, high, low, close
#         from daily_bars
#         where stock_id = (select id from stock where UPPER(symbol) = %s)
#         order by day asc""", connection, params=(symbol.upper(),))
#
#     st.subheader(symbol.upper())
#
#     fig = go.Figure(data=[go.Candlestick(x=data['day'],
#                     open=data['open'],
#                     high=data['high'],
#                     low=data['low'],
#                     close=data['close'],
#                     name=symbol)])
#
#     fig.update_xaxes(type='category')
#     fig.update_layout(height=700)
#
#     st.plotly_chart(fig, use_container_width=True)
#
#     st.write(data)
#
#
# if option == 'wallstreetbets':
#     num_days = st.sidebar.slider('Number of days', 1, 30, 3)
#
#     cursor.execute("""
#         SELECT COUNT(*) AS num_mentions, symbol
#         FROM mention JOIN stock ON stock.id = mention.stock_id
#         WHERE date(dt) > current_date - interval '%s day'
#         GROUP BY stock_id, symbol
#         HAVING COUNT(symbol) > 10
#         ORDER BY num_mentions DESC
#     """, (num_days,))
#
#     counts = cursor.fetchall()
#     for count in counts:
#         st.write(count)
#
#     cursor.execute("""
#         SELECT symbol, message, url, dt, username
#         FROM mention JOIN stock ON stock.id = mention.stock_id
#         ORDER BY dt DESC
#         LIMIT 100
#     """)
#
#     mentions = cursor.fetchall()
#     for mention in mentions:
#         st.text(mention['dt'])
#         st.text(mention['symbol'])
#         st.text(mention['message'])
#         st.text(mention['url'])
#         st.text(mention['username'])
#
#     rows = cursor.fetchall()
#
#     st.write(rows)
#
#
# if option == 'pattern':
#     pattern = st.sidebar.selectbox(
#         "Which Pattern?",
#         ("engulfing", "threebar")
#     )
#
#     if pattern == 'engulfing':
#         cursor.execute("""
#             SELECT *
#             FROM (
#                 SELECT day, open, close, stock_id, symbol,
#                 LAG(close, 1) OVER ( PARTITION BY stock_id ORDER BY day ) previous_close,
#                 LAG(open, 1) OVER ( PARTITION BY stock_id ORDER BY day ) previous_open
#                 FROM daily_bars
#                 JOIN stock ON stock.id = daily_bars.stock_id
#             ) a
#             WHERE previous_close < previous_open AND close > previous_open AND open < previous_close
#             AND day = '2021-02-18'
#         """)
#
#     if pattern == 'threebar':
#         cursor.execute("""
#             SELECT *
#             FROM (
#                 SELECT day, close, volume, stock_id, symbol,
#                 LAG(close, 1) OVER ( PARTITION BY stock_id ORDER BY day ) previous_close,
#                 LAG(volume, 1) OVER ( PARTITION BY stock_id ORDER BY day ) previous_volume,
#                 LAG(close, 2) OVER ( PARTITION BY stock_id ORDER BY day ) previous_previous_close,
#                 LAG(volume, 2) OVER ( PARTITION BY stock_id ORDER BY day ) previous_previous_volume,
#                 LAG(close, 3) OVER ( PARTITION BY stock_id ORDER BY day ) previous_previous_previous_close,
#                 LAG(volume, 3) OVER ( PARTITION BY stock_id ORDER BY day ) previous_previous_previous_volume
#             FROM daily_bars
#             JOIN stock ON stock.id = daily_bars.stock_id) a
#             WHERE close > previous_previous_previous_close
#                 AND previous_close < previous_previous_close
#                 AND previous_close < previous_previous_previous_close
#                 AND volume > previous_volume
#                 AND previous_volume < previous_previous_volume
#                 AND previous_previous_volume < previous_previous_previous_volume
#                 AND day = '2021-02-19'
#         """)
#
#     rows = cursor.fetchall()
#
#     for row in rows:
#         st.image(f"https://finviz.com/chart.ashx?t={row['symbol']}")
#
#
# if option == 'stocktwits':
#     symbol = st.sidebar.text_input("Symbol", value='AAPL', max_chars=5)
#
#     r = requests.get(f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json")
#
#     data = r.json()
#
#     for message in data['messages']:
#         st.image(message['user']['avatar_url'])
#         st.write(message['user']['username'])
#         st.write(message['created_at'])
#         st.write(message['body'])

import pandas as pd  #pip install pandas openpyxl


dfTrades = pd.read_excel(
	io='dfTrades.xlsx',
	engine='openpyxl',
	sheet_name='Sheet1',
	skiprows=0,
	usecols='A:L',
	nrows=36,
)

print(df)

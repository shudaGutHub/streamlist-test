import pandas as pd  #pip install pandas openpyxl


df = pd.read_excel(
	io='dfCapacityCalls_BDIN_Portfolio_2021-09-10.csv',
	engine='openpyxl',
	sheet_name='dfCapacityCalls_BDIN_Portfolio_',
	skiprows=0,
	usecols='B:W',
	nrows=38,
)

print(df)

import pandas as pd  #pip install pandas openpyxl


df = pd.read_excel(
	io='dfDashSummary_2021-09-10.csv',
	engine='openpyxl',
	sheet_name='dfDashSummary_2021-09-10',
	skiprows=0,
	usecols='B:AH',
	nrows=174,
)

print(df)

import pandas as pd  #pip install pandas openpyxl


df = pd.read_excel(
	io='dfBD.xlsx',
	engine='openpyxl',
	sheet_name='Sheet1',
	skiprows=0,
	usecols='A:AT',
	nrows=190,
)

print(df)

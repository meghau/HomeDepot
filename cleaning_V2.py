import pandas

data = pandas.read_csv('clean_data.csv')
data = data.dropna()
data = data.reset_index()

data.to_csv('clean_data.csv')
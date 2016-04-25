import pandas

#drops NA values that are created after cleaning
data = pandas.read_csv('clean_data.csv')
data = data.dropna()
data = data.reset_index()

data.to_csv('clean_data.csv')
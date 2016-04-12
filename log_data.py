# code to load data and merge 2 datasets

import pandas as pd

# set path variable to the path of datasets
path = "/home/manish/ADGBI/Capstone/Data/"

prod_desc = pd.read_csv(path + "product_descriptions.csv")

train = pd.read_csv(path + "train.csv")

data = pd.merge(train,prod_desc)

data.to_csv(path + "data.csv")
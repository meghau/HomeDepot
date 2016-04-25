# code to load data and merge 2 datasets

import pandas as pd

prod_desc = pd.read_csv("product_descriptions.csv")

train = pd.read_csv("train.csv")

data = pd.merge(train,prod_desc)

data.to_csv("data.csv")
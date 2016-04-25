from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, AdaBoostRegressor

train = pd.read_csv('data_train.csv')
test = pd.read_csv('data_test.csv')

y_train = train['relevance'].values
X_train = train.drop(['relevance'],axis=1).values
y_test = test['relevance'].values
X_test = test.drop(['relevance'],axis=1).values

# K-Nearest Neighbor

knr=KNeighborsRegressor(n_neighbors=500)
#0.5269
knr.fit(X_train, y_train)
y_pred = knr.predict(X_test)

# confusion_matrix(y_test, y_pred)
rms = sqrt(mean_squared_error(y_test, y_pred))
print rms












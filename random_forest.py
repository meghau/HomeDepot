from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, AdaBoostRegressor
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt

# loading the train and test data
train = pd.read_csv('data_train.csv')
test = pd.read_csv('data_test.csv')

y_train = train['relevance'].values
X_train = train.drop(['relevance'],axis=1).values
y_test = test['relevance'].values
X_test = test.drop(['relevance'],axis=1).values

# Tried adaboost, but not better than Random forest
# ad = AdaBoostRegressor(base_estimator=None, n_estimators=18, learning_rate=0.5, loss='linear', random_state=0)

# random forest regressor model
rf = RandomForestRegressor(n_estimators=15, max_depth=6, random_state=0)

# bagging of random forest regressor  
br = BaggingRegressor(rf, n_estimators=45, max_samples=0.1, random_state=25)

# building the model 
br.fit(X_train, y_train)

# predicting the y - values using the model built
y_pred = br.predict(X_test)

# calculating the root mean square error
rms = sqrt(mean_squared_error(y_test, y_pred))
print rms

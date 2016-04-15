from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, AdaBoostRegressor
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt

train = pd.read_csv('data_train.csv')
test = pd.read_csv('data_test.csv')

y_train = train['relevance'].values
X_train = train.drop(['id','relevance'],axis=1).values
y_test = test['relevance'].values
X_test = test.drop(['id','relevance'],axis=1).values

# Tried adaboost, but not better than Random forest
# ad = AdaBoostRegressor(base_estimator=None, n_estimators=18, learning_rate=0.5, loss='linear', random_state=0)

rf = RandomForestRegressor(n_estimators=15, max_depth=6, random_state=0)
clf = BaggingRegressor(rf, n_estimators=45, max_samples=0.1, random_state=25)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# confusion_matrix(y_test, y_pred)
rms = sqrt(mean_squared_error(y_test, y_pred))
print rms

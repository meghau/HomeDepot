import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt

data = pd.read_csv("clean_data.csv")
desc = data['product_description']
term = data['search_term']
Y = np.asarray(data['relevance'])

query = []
for q in term:
	query.append(q.split(" "))

tf = TfidfVectorizer(analyzer='word', ngram_range=(1,3), min_df = 0, stop_words = 'english')
vec_desc = tf.fit_transform(desc)
feat_names_desc = tf.get_feature_names()
l = len(tf.idf_)
vec_query = []
for q in query:
	q_vec = [0]*l
	for w in q:
		if w in feat_names_desc:
			i = feat_names_desc.index(w)
			q_vec[i] = tf.idf_[i]
	vec_query.append(q_vec)

vec_desc_dense = vec_desc.todense()
vec_desc_array = []
for m in vec_desc_dense:
	vec_desc_array.append(np.squeeze(np.asarray(m)))

X = []
for i in range(len(desc)):
	X.append(cosine_similarity(vec_desc_array[i], vec_query[i])[0][0])
X = np.asarray(X)
X = X.reshape((len(X), 1))
#import code; code.interact(local=locals())
model = SGDRegressor()
model.fit(X, Y)

y_pred = model.predict(X)
rms = sqrt(mean_squared_error(Y, y_pred))
print rms
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
import pandas as pd
import numpy as np

data = pd.read_csv("clean_data.csv")
desc = data['product_description'][:5000]
term = data['search_term'][:5000]
Y = np.asarray(data['relevance'][:5000])

query = [q.split(" ") for q in term]

tf = TfidfVectorizer(analyzer='word', ngram_range=(1,1), min_df = 0, stop_words = 'english')
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

vec_desc_array = [np.squeeze(np.asarray(m)) for m in vec_desc.todense()]
X = [cosine_similarity(vec_desc_array[i], vec_query[i])[0][0] for i in range(len(desc))]
X = np.asarray(X)
X = X.reshape((len(X), 1))
#import code; code.interact(local=locals())
model = SGDRegressor()
model.fit(X, Y)

y_pred = model.predict(X)
rms = sqrt(mean_squared_error(Y, y_pred))
print rms
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from gensim import corpora, models, similarities
from math import sqrt
import pandas as pd
import numpy as np
import csv

data = pd.read_csv("clean_data.csv")
desc = data['product_description']
term = data['search_term']
Y = np.asarray(data['relevance'])

class MyCorpus(object):
	def __iter__(self):
		for line in desc:
			yield dictionary.doc2bow(line.lower().split())

dictionary = corpora.Dictionary(line.lower().split() for line in desc)
corpus = MyCorpus()
tfidf = models.TfidfModel(corpus)
lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=50)
#import code;code.interact(local=locals())
query = [str(q).split(" ") for q in term]
desc = [d.split(" ") for d in desc]



S = []
L = []
for i in range(len(desc)):
	S.append(similarities.MatrixSimilarity([tfidf[dictionary.doc2bow(desc[i])]],num_features=len(dictionary))[tfidf[dictionary.doc2bow(query[i])]])
	L.append(similarities.MatrixSimilarity([lsi[tfidf[dictionary.doc2bow(desc[i])]]],num_features=len(dictionary))[lsi[tfidf[dictionary.doc2bow(query[i])]]])

#import code;code.interact(local=locals())
A = [s[0] for s in S]
B = [l[0] for l in L]
d = pd.DataFrame({'S':pd.Series(A), 'L':pd.Series(B)})
d.to_csv('vsmlsi.csv',header=False)
A = np.asarray(A)
B = np.asarray(B)

X = np.column_stack((A, B))
#X = X.reshape((len(X), 1))
model = SGDRegressor()
model.fit(X, Y)
y_pred = model.predict(X)
rms = sqrt(mean_squared_error(Y, y_pred))
print rms

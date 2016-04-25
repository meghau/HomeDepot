from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from gensim import corpora, models, similarities
from math import sqrt
import pandas as pd
import numpy as np
import csv

#Read relevant columns from clean data
data = pd.read_csv("clean_data.csv")
desc = data['product_description']
term = data['search_term']
Y = np.asarray(data['relevance'])

#Generating corpus
class MyCorpus(object):
	def __iter__(self):
		for line in desc:
			yield dictionary.doc2bow(line.lower().split())

#Generate dictionary
dictionary = corpora.Dictionary(line.lower().split() for line in desc)
corpus = MyCorpus()

#Generate tf-idf model of words
tfidf = models.TfidfModel(corpus)

#Generate lsi model from corpus
lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=50)

#Split words in query and description
query = [str(q).split(" ") for q in term]
desc = [d.split(" ") for d in desc]



S = []
L = []
#Calculate cosine similarity between each query and description
for i in range(len(desc)):
	S.append(similarities.MatrixSimilarity([tfidf[dictionary.doc2bow(desc[i])]],num_features=len(dictionary))[tfidf[dictionary.doc2bow(query[i])]])
	L.append(similarities.MatrixSimilarity([lsi[tfidf[dictionary.doc2bow(desc[i])]]],num_features=len(dictionary))[lsi[tfidf[dictionary.doc2bow(query[i])]]])

#Write the two feature columns to csv file
A = [s[0] for s in S]
B = [l[0] for l in L]
d = pd.DataFrame({'S':pd.Series(A), 'L':pd.Series(B)})
d.to_csv('vsmlsi.csv',header=False)
A = np.asarray(A)
B = np.asarray(B)

#Pass the two features to SGDRegressor
X = np.column_stack((A, B))
#X = X.reshape((len(X), 1))
model = SGDRegressor()
model.fit(X, Y)

#Predict the relevance
y_pred = model.predict(X)

#Calculate rmse
rms = sqrt(mean_squared_error(Y, y_pred))
print rms

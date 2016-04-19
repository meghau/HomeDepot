import pandas as pd
from gensim.models.doc2vec import LabeledSentence, Doc2Vec
import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

data = pd.read_csv("clean_data.csv")
#data = pd.read_csv("/home/manish/ADGBI/Capstone/HomeDepot/data.csv")



data_pd = data['product_description']
data_st = data['search_term']

# create labeled sentence for data_pd, data_st

def labeled_sentence(x,y):
    sentence = []
    for i in range(len(x)):
        tag = y + str(i)
        sentence.append(LabeledSentence(words=x[i], tags=[tag]))
    return sentence

def extract(m,x,c):
    f = []
    for i in range(c):
        tag = x + str(i)
        f.append(m.docvecs[tag])
    return f

labeled_pd = labeled_sentence(data_pd,'pd_')

model_pd = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=4)
model_pd.build_vocab(labeled_pd)

for i in range(5):
        print "Training iteration %d" % (i)
        random.shuffle(labeled_pd)
        model_pd.train(labeled_pd)

data_pd_vec = extract(model_pd,'pd_',len(data_pd))

labeled_st = labeled_sentence(data_st,'st_')

model_st = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=4)
model_st.build_vocab(labeled_st)

for i in range(5):
        print "Training iteration %d" % (i)
        random.shuffle(labeled_st)
        model_st.train(labeled_st)

data_st_vec = extract(model_st,'st_',len(data_st))

df = pd.Series()

def cosinesim(a, b):
    return np.dot(a,b) / np.sqrt(np.dot(a, a)) / np.sqrt(np.dot(b, b))

for i in range(len(data)):
    df = df.set_value(i,cosinesim(data_st_vec[i],data_pd_vec[i]))

min_val = df.min()
maxmin_val = df.max() - min_val

for i in range(len(data)):
    df = df.set_value(i,(df[i]-min_val)/maxmin_val)

df.to_csv("doc2vec_feature.csv")

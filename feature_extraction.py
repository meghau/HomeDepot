import pandas as pd
import numpy as np

np.random.seed(50)

# to convert each column into a vector of words
def vectorize(data):
	vectorized_dataset = data.copy(deep = True)
	for col_name in ('product_title', 'product_description', 'search_term'):
		vectorized_dataset[col_name] = data[col_name].map(lambda x: str(x).lower().split(' '))
	return vectorized_dataset


def extract_features(v_data):
	data_features = v_data.copy(deep = True)
	# title - search term match
	data_features['title_search_term_match'] = data_features.apply(lambda x: len(set(x['product_title']).intersection(x['search_term'])), axis = 1)
	# desc - search term match
	data_features['desc_search_term_match'] = data_features.apply(lambda x: len(set(x['product_description']).intersection(x['search_term'])), axis = 1)
	# length of search term
	data_features['length_search_term'] = data_features.apply(lambda x: len(x['search_term']), axis = 1)
	
	# last word of title and search term match (the actual product)
	def last_word_match(x):
		if x['search_term'][-1] == x['product_title'][-1]:
			return 1
		else:
			return 0

	data_features['last_word_match'] = data_features.apply(lambda x: last_word_match(x) , axis = 1)

	# assuming brand names are at the beginning of product title for some products
	def brand_name_match(x):
		if x['product_title'][0] in x['search_term']:
			return 1
		else:
			return 0

	# brand name match
	data_features['brand_name_match'] = data_features.apply(lambda x: brand_name_match(x) , axis = 1)

	# ratio of the number of matches between product title and search term and the number of words in the search term
	data_features['ratio_title']=data_features['title_search_term_match']/data_features['length_search_term']

	# ratio of the number of matches between product description and search term and the number of words in the search term
	data_features['ratio_description']=data_features['desc_search_term_match']/data_features['length_search_term']

	# integrating vector space model features
	vsm_data = pd.read_csv("vsm.csv", header = None)
	data_features['vsm'] = vsm_data[1]

	# integrating doc2vec model features
	doc2vec_data = pd.read_csv("doc2vec_feature.csv", header = None)
	data_features['doc2vec'] = doc2vec_data[1]

	data_features = data_features.drop(['Unnamed: 0','Unnamed: 0.1','Unnamed: 0.1','id','index','search_term','product_title','product_description'],axis=1)
	return data_features
	
def main():
	dataset = pd.read_csv("clean_data.csv")
	# removing NA from Dataset
	# dataset = dataset.dropna()
	
	vectorized_dataset = vectorize(dataset)
	featurized_dataset = extract_features(vectorized_dataset)
	
	# dividing the dataset into train and test
	index = np.random.rand(len(dataset)) < 0.7
	train = featurized_dataset[index]
	test = featurized_dataset[~index]

	# storing as csv
	train.to_csv('data_train.csv',index=False)
	test.to_csv('data_test.csv',index=False)
	
if __name__ == "__main__":
    main()

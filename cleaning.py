import pandas
from nltk import*
from nltk.corpus import wordnet
import re
import nltk.data
import nltk.tag
import requests
import sys
import time
from random import randint
download("stopwords")

#removes jointwords
def remove_jointwords(word):
	clean_list=[]
 	for s in word.split(" "):
    		for i,word in enumerate(s):
        		if not word.isupper() and not word.islower() and word!=s[0]:
            			s=s[:i]+' '+s[i:]
            			break
            	clean_list.append(s.split(" "))           			
	flattened_list=[item for sublist in clean_list for item in sublist]
	str1 = ' '.join(flattened_list)
	return str1

#POS tagging of the words
def word_tag(tag):
	if tag.startswith('J'):
		return wordnet.ADJ
	elif tag.startswith('V'):
		return wordnet.VERB
	elif tag.startswith('N'):
		return wordnet.NOUN
	elif tag.startswith('R'):
		return wordnet.ADV
	else:
		return ''

#cleans the data
def clean_entry(entry):
	entry = remove_jointwords(entry)
	#changes to lowercase
	entry = entry.lower()
	entry = word_tokenize(entry)
	list_words = tagger.tag(entry)
	entry = list()
	for tup in list_words:
		word = tup[0]
		tag = word_tag(tup[1])
		if tag!='':
			lemmatized_word = lemmatizer.lemmatize(word,tag)
			lemmatized_word = lemmatized_word.lower()
			entry.append(lemmatized_word)
	entry = ' '.join(entry)
	entry = re.sub("[^\w]", " ",  entry).split()
	entry = ' '.join(entry)
	print entry
	return entry

def main():
	global prd_entry,stopwords,lemmatizer,tagger
	tagger = PerceptronTagger()
	lemmatizer = WordNetLemmatizer()
	data = pandas.read_csv('data.csv')
	#removes stopwords
	stopwords = set(corpus.stopwords.words('english'))

	#reads data from data.csv
	data['product_title'] = data['product_title'].map(lambda x:clean_entry(x))
	data['product_title'] = data['product_title'].map(lambda x: set(word_tokenize(x)) - stopwords).map(lambda l:' '.join(l))

	data['product_description'] = data['product_description'].map(lambda x:clean_entry(x))
	data['product_description'] = data['product_description'].map(lambda x: set(word_tokenize(x)) - stopwords).map(lambda l:' '.join(l))
	
	data['search_term'] = data['search_term'].map(lambda x:clean_entry(x))
	data['search_term'] = data['search_term'].map(lambda x: set(word_tokenize(x)) - stopwords).map(lambda l:' '.join(l))

	data = data.dropna()

	#writes the data to new csv file
	data.to_csv('clean_data.csv')
	

if __name__ == "__main__":
    main()
	
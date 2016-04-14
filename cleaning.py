import pandas
from nltk import*
from nltk.corpus import wordnet
import re
#import text2num

#download("stopwords")
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


def clean_entry(entry):
	entry = entry.lower()
	entry = word_tokenize(entry)
	list_words = list()
	for word in entry:
		list_words.append(stemmer.stem(word))
	entry = ' '.join(list_words)
	entry = re.sub("[^\w]", " ",  entry).split()
	entry = ' '.join(entry)
	return entry


def main():
	global prd_entry,stopwords,lemmatizer,stemmer
	data = pandas.read_csv('data.csv')
	stopwords = set(corpus.stopwords.words('english'))
	stemmer = SnowballStemmer('english')
	#data_subset = data[['product_uid','product_title','product_description']]
	#data_subset = data_subset.drop_duplicates()
	data['product_title'] = data['product_title'].map(lambda x:clean_entry(x))
	data['product_title'] = data['product_title'].map(lambda x: set(word_tokenize(x)) - stopwords).map(lambda l:' '.join(l))

	data['product_description'] = data['product_description'].map(lambda x:clean_entry(x))
	data['product_description'] = data['product_description'].map(lambda x: set(word_tokenize(x)) - stopwords).map(lambda l:' '.join(l))

	data['search_term'] = data['search_term'].map(lambda x:clean_entry(x))
	data['search_term'] = data['search_term'].map(lambda x: set(word_tokenize(x)) - stopwords).map(lambda l:' '.join(l))

	print data['product_title'][0]
	print data['product_description'][0]
	print data['search_term'][0]

	#data_subset['product_title'].apply(clean_entry)
	#data_subset['product_description'].apply(clean_entry)
	data.to_csv('clean_data.csv')
	


if __name__ == "__main__":
    main()
	
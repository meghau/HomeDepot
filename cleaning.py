import pandas
from nltk import*
from nltk.corpus import wordnet
import re
import nltk.data
import nltk.tag

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
	list_words = tagger.tag(entry)
	entry = list()
	for tup in list_words:
		word = tup[0]
		tag = word_tag(tup[1])
		if tag!='':
			entry.append(lemmatizer.lemmatize(word,tag))
	entry = ' '.join(entry)
	entry = re.sub("[^\w]", " ",  entry).split()
	entry = ' '.join(entry)
	return entry


def main():
	global prd_entry,stopwords,lemmatizer,stemmer,tagger
	tagger = PerceptronTagger()
	lemmatizer = WordNetLemmatizer()
	data = pandas.read_csv('data.csv')
	stopwords = set(corpus.stopwords.words('english'))
	stemmer = SnowballStemmer('english')
	data['product_title'] = data['product_title'].map(lambda x:clean_entry(x))
	data['product_title'] = data['product_title'].map(lambda x: set(word_tokenize(x)) - stopwords).map(lambda l:' '.join(l))

	data['product_description'] = data['product_description'].map(lambda x:clean_entry(x))
	data['product_description'] = data['product_description'].map(lambda x: set(word_tokenize(x)) - stopwords).map(lambda l:' '.join(l))

	data['search_term'] = data['search_term'].map(lambda x:clean_entry(x))
	data['search_term'] = data['search_term'].map(lambda x: set(word_tokenize(x)) - stopwords).map(lambda l:' '.join(l))

	data = data.dropna()
	data.to_csv('clean_data.csv')
	

if __name__ == "__main__":
    main()
	
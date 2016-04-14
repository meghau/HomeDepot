import pandas
from nltk import*
from nltk.corpus import wordnet
import re
import text2num

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
	tokens_pos = pos_tag(entry)
	entry = list()
	for token in tokens_pos:
		tag = word_tag(token[1])
		if tag!='':
			entry.append(lemmatizer.lemmatize(token[0],tag))
	entry = ' '.join(entry)
	entry_wordList = re.sub("[^\w]", " ",  entry).split()
	entry = ' '.join((set(entry_wordList) - stopwords))
	print entry
	return entry


def main():
	global prd_entry,stopwords,lemmatizer
	data = pandas.read_csv('data.csv')
	stopwords = set(corpus.stopwords.words('english'))
	lemmatizer = WordNetLemmatizer()
	data_subset = data[['product_uid','product_title','product_description']]
	data_subset = data_subset.drop_duplicates()
	data_subset['product_title'] = data_subset['product_title'].map(lambda x: set(word_tokenize(x)) - stopwords).map(lambda l:''.join(l))
	print data_subset['product_title'][0]

	#data_subset['product_title'].apply(clean_entry)
	#data_subset['product_description'].apply(clean_entry)
	#data_subset.to_csv('clean_data.csv')
	


if __name__ == "__main__":
    main()
	
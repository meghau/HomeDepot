import requests
import sys
import time
from random import randint
import pandas

# take arg as the entire search_term column
data = pandas.read_csv('data.csv')
search_term = data['search_term']

# gas mowe -> gas+mowe
# the txed format is used in google search request
def split(a):
    a = a.split()
    a = '+'.join(map(str,a))
    return a

# apply split to all rows in search_term
search_term = search_term.apply(split)

# values used to find relevant information in text obtained after requests.get
showing_results_for="<span class=\"spell\">Showing results for</span>"
search_instead_for="<br><span class=\"spell_orig\">Search instead for"

# function to correct spelling

def spell_check(st):
    # suggested
    time.sleep(randint(0,2) )
    # get data for search query
    response = requests.get("https://www.google.com/search?q="+st)
    # determine where the text needs to be found
    start = response.text.find(showing_results_for)
    end = response.text.find(search_instead_for)
    if(start>-1):
        resp_text = response.text[start:end]
        i = str(resp_text).find('/search?q=')
        j = resp_text[i:].find('&amp')
        search = str(resp_text[i+10:i+j])
    else:
        search = st
    return search;

# apply spell_check to split data
search_term = search_term.apply(spell_check)

# gas+mower -> gas mower
def merge(a):
    a = a.replace('+',' ')
    return a

# apply merge to spell checked data
search_term = search_term.apply(merge)

data.to_csv('data.csv')

# ignore below comments
# used while working on cmd line
# st = data['search_term'][0:20]
# st = st.apply(split)
# st = st.apply(spell_check)

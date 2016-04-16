import sys
import collections
import nltk
import operator
import numpy as np


def main():
    remove_jointwords("SunRise HomeWork")
    

def remove_jointwords(desc):
	clean_list=[]
 	for s in desc.split(" "):
    		for i,word in enumerate(s):
        		if word.isupper() and word!=s[0]:
            			s=s[:i]+' '+s[i:]
            			clean_list.append(s.split(" "))
            			break
	flattened_list=[item for sublist in clean_list for item in sublist]
	str1 = ' '.join(flattened_list)
	return str1

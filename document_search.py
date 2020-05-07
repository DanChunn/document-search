import os
import logging
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import string
import re
import numpy as np
from datetime import datetime
import locale
import timeit

nltk.download('stopwords')
nltk.download('punkt')

debug = False
data_folder = os.getcwd() + "/document-search/data/"

data_set = {} #{title:text}
processed_data_set = {} #dataset of titles and their word tdidfs  {title : {word : {tf, idf, tfidf}}

def to_lower_case(text):
	return text.lower()

def replace_commas(text):
	#quick fix for numbers like 2,000, could use regex?
	return text.replace(',','')

#removes stop words AND words that are of single character length
def remove_stop_words(text):
	stop_words = stopwords.words('english')
	words = word_tokenize(text)
	processed_text = ""
	for w in words:
			if len(w) > 1 and w not in stop_words:
					processed_text = processed_text + " " + w 
	return processed_text


def remove_punctuation(text):
	translator = str.maketrans(string.punctuation, ' '*len(string.punctuation)) #map punctuation to space
	return text.translate(translator)


def stemming(text):
    stemmer= PorterStemmer()
    words = word_tokenize(str(text))
    processed_text = ""
    for w in words:
        processed_text = processed_text + " " + stemmer.stem(w)
    return processed_text

def preprocess(text, title=""):
	text = to_lower_case(text)
	text = replace_commas(text)
	text = remove_punctuation(text)
	text = remove_stop_words(text) #expensive
	text = stemming(text) #expensive
	return text

def preprocess_query(text, title=""):
	text = to_lower_case(text)
	text = replace_commas(text)
	text = remove_punctuation(text)
	text = stemming(text) #expensive
	return text

def read_data(path=data_folder):
	print('Reading data...')
	start_time = datetime.now()

	extensions = ['.txt']
	for subdir, dirs, files in os.walk(path):
		for name in files:
			if name.endswith(tuple(extensions)):
				raw_title = name

				with open(path+name, 'r') as file:
					raw_text = file.read().replace('\n', '')
					data_set[raw_title] = raw_text

	end_time = datetime.now()
	duration = end_time - start_time
	print('Reading data finished in {}'.format(str(duration.total_seconds()*1000) + ' ms'))


#processes entire dataset, counts number of terms found within each document
def process_corpus(data=data_set):
	print('Preprocessing data...')
	start_time = datetime.now()
	unique_word_count_in_corpus = {} #words found atleast once in documents across corpus {word:count}

	for title in data_set:
		text = data_set[title]

		processed_text = preprocess(text, title).split(' ')
		#total_word_count = len(processed_text)
		max_term_count = 0
		processed_text_dict = {}
		
		#get tf for each word
		for w in processed_text:
			if w in processed_text_dict:
				processed_text_dict[w] += 1
				if processed_text_dict[w] > max_term_count:
					max_term_count = processed_text_dict[w]
			else: 
				processed_text_dict[w] = 1
				if w in unique_word_count_in_corpus:
					unique_word_count_in_corpus[w] += 1
				else:
					unique_word_count_in_corpus[w] = 1

		#convert tf into scores based on tf / count of most popular word
		for word, tf in processed_text_dict.items():
			processed_text_dict[word] = tf / max_term_count  #dampening
		
		processed_text_dict.pop('', None)
		processed_data_set[title] = processed_text_dict

	#get idf and tfidf for each word
	for word in unique_word_count_in_corpus:
		idf = len(data_set) / unique_word_count_in_corpus[w]
		#processed_data_set: {title : {word : tf_score, word : tf_score}}
		for title in processed_data_set:
			data = processed_data_set[title] 
			if word in data:
				tf = data[word]
				tfidf_data = {}
				tfidf_data['tf'] = round(tf,6)
				tfidf_data['idf'] = round(idf,6)
				tfidf_data['tfidf'] = round(tf*idf,5)
				data[word] = tfidf_data

	#print(processed_data_set)
	end_time = datetime.now()
	duration = end_time - start_time
	print('Preprocessing finished in {}'.format(str(duration.total_seconds()*1000) + ' ms'))
	logging.info(processed_data_set)


#SEARCH FUNCTIONS
#search by regex on raw unedited text
def search_regex(regex):
	logging.info(' --- regex ---')
	print('\nregex search query - {}'.format(regex))

	res = {}
	for title in data_set:
		text = data_set[title]

		found = re.findall(regex, text)
		res[title] = len(found)
		logging.info('RESULTS FROM {} --- {}'.format(title, found))

	logging.info(' ------------\n')
	return sorted(res.items(), key=lambda x:x[1], reverse=True)

#search by exact string match on raw unedited text, case sensitive
def search_simple(query):
	logging.info(' --- match ---')
	query = query
	print('\nsimple search query - {}'.format(query))

	res = {}
	for title in data_set:
		text = data_set[title]

		found = re.findall(query, text, flags=re.IGNORECASE)
		res[title] = len(found)
		logging.info('RESULTS FROM {} --- {}'.format(title, found))

	logging.info(' ------------\n')
	return sorted(res.items(), key=lambda x:x[1], reverse=True)



#search using by term frequency score
def search_index(query):
	logging.info(' --- index ---')
	print('\nindex search query - {}'.format(query))
	res = {}
	query = preprocess_query(query, 'query')
	query = set(query.split(' '))
	if '' in query:
		query.remove('')
	print('index search processed query - {}'.format(query))
	# print(processed_data_set)

	for query_term in query:
		for title in processed_data_set:
			text_data = processed_data_set[title]

			if query_term in text_data:
				tfidf_score = text_data[query_term]['tfidf']
				if title in res:
					res[title] += tfidf_score
				else:
					res[title] = tfidf_score

	logging.info(' ------------\n')
	return sorted(res.items(), key=lambda x:x[1], reverse=True)


class CodeTimer:
    def __init__(self, name=None):
        self.name = " '"  + name + "'" if name else ''

    def __enter__(self):
        self.start = timeit.default_timer()

    def __exit__(self, exc_type, exc_value, traceback):
        self.took = (timeit.default_timer() - self.start) * 1000.0
        print('Code block' + self.name + ' took: ' + str(self.took) + ' ms')



debug = False
if(debug):
	logging.getLogger().setLevel(logging.INFO)

print('\n\n')
read_data()
process_corpus()
print('\n')

# SET TO TRUE IF YOU WISH TO RUN BY COMMAND LINE
running = True
while(running):
	query = input("\nEnter your search query:\n") 
	search_type = input("\nEnter search type: (match / regex / index)\n").lower()

	print('\nYour query is \"{}\" using {} search!\n'.format(query, search_type))

	if search_type == 'match':
		print('\n'+ str(search_simple(query))+'\n')
	elif search_type == 'regex':
		print('\n'+ str(search_regex(query))+'\n')
	elif search_type == 'index':
		print('\n'+ str(search_index(query))+'\n')
	elif search_type == 'stop':
		running = False
	else:
		search_type == None
		print('Invalid search type... restarting query')
		continue



#increase range for stress testing
with CodeTimer('match loop'):
	count = 0
	for i in range(0,1):
		print('...')
		print(search_simple('warp'))
		print(search_simple('hitchhiker'))
		print(search_simple('french war'))
		print(search_simple('the'))
		print(search_simple('travel'))
		print(search_simple('Following defeat in the Franco-Prussian War'))
		print(search_simple('follower'))

print('\n')

with CodeTimer('regex loop'):
	count = 0
	for i in range(0,1):
		print('...')
		print(search_regex(r'warp'))
		print(search_regex(r'hitchhiker'))
		print(search_regex(r'french war'))
		print(search_regex(r'the'))
		print(search_regex(r'travel'))
		print(search_regex(r'Following defeat in the Franco-Prussian War'))
		print(search_regex(r'follower'))


		#print(search_regex('([A-Z][a-z]+)') )
		#print(search_regex('([A-Z]+rench)'))
		#print(search_regex('([F]+)'))
print('\n')

with CodeTimer('index loop'):
	count = 0
	for i in range(0,1):
		print('...')
		print(search_index('warp')) 
		print(search_index('hitchhiker'))
		print(search_index('french war'))
		print(search_index('the'))
		print(search_index('travel'))
		print(search_index('Following defeat in the Franco-Prussian War'))
		print(search_index('follower'))

#print(processed_data_set)
# Basics
from __future__ import print_function
import pandas as pd
import numpy as np
import os
import re
import json
from itertools import chain
from time import time
import operator
import math
import numpy as np
import string
import matplotlib.pyplot as plt
# NLTK / NLP libraries
from nltk.tag import pos_tag
from nltk.util import ngrams
from nltk.tokenize import word_tokenize, wordpunct_tokenize, WhitespaceTokenizer, sent_tokenize, MWETokenizer
from nltk.corpus import stopwords
from nltk.chunk import ne_chunk
from nltk.stem.wordnet import WordNetLemmatizer 
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.porter import PorterStemmer
from nltk import pos_tag
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from stop_words import get_stop_words
from gensim import corpora, models, similarities, matutils
from textblob import TextBlob
#sklearn
from sklearn.decomposition import TruncatedSVD
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn import preprocessing
from sklearn.cluster import KMeans, DBSCAN
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import linear_model
from sklearn import ensemble
from sklearn import model_selection
from sklearn.naive_bayes import MultinomialNB, GaussianNB
import pyLDAvis.gensim

def load_dataframe(filepath):
	''' 
	Function to load csv file
	INPUT: csv file which was saved from mongodb.py
	OUTPUT: Dataframe with all comments sorted by user
	'''

	df = pd.read_csv(filepath)
	# Remove users tagged 'deleted'
	df = df[df['author'] != '[deleted]'].dropna()
	# Sort users alphabetically
	df = df.sort_values('author')
	return df

def lemmatize(sentence):
	''' 
	Function to lemmatize a sentence
	INPUT: Sentence
	OUTPUT: Lemmatized sentence
	'''

    lem = WordNetLemmatizer()
    lemmed = [lem.lemmatize(w.replace(',','').replace('"','')) for w in sentence.split(' ')]
    
    return " ".join(lemmed)

def translate_slang(sentence):
	''' 
	Developable function to translate slang
	INPUT: Sentence
	OUTPUT: Sentence without slang
	'''

    slang = {'rt':'Retweet', 'dm':'direct message', 'awsm' : 'awesome', 'luv' :'love'}
    std = []
    for word in sentence.split(' '):
        if word in slang:
            word = slang[word.lower()]
            std.append(word)
        else:
            std.append(word)

    return " ".join(std)

def sentiment(sentence):
	''' 
	Function to analyze the sentiment of a sentence
	INPUT: Sentence
	OUTPUT: Sentiments of each word in the sentence
	'''

    sid = SentimentIntensityAnalyzer()
    sent = []
    for word in sentence.split(' '):
        #print(word)
        token = sent_tokenize(word)
        sentence_stats = map(lambda sentence: sid.polarity_scores(sentence), token)
        stats = {'neg': 0, 'neu': 0, 'pos': 0}
        for stat in sentence_stats:
            stats['neg'] += stat['neg']
            stats['neu'] += stat['neu']
            stats['pos'] += stat['pos']
        len_sentences_stats = len(list(sentence_stats))
        if len_sentences_stats != 0:
            stats['neg'] /= len_sentences_stats
            stats['neu'] /= len_sentences_stats
            stats['pos'] /= len_sentences_stats
        sent.append([stats['neg'],stats['neu'],stats['pos']])
    return sent

def clean_text(X):
	''' 
	Function to clean column with comments in the dataframe
	INPUT: Dataframe -> author, controversiality, score, text
	OUTPUT: Cleaned dataframe
	'''

    # list of stop words
    stop = set(['Heh','heh','language','know', 'wow', 'hah', 'hey','really','year', 'yeah','wtf', 'meh', 'oops', 'nah', 'yea','doesnt','dont','make',
        'huh', 'mar', 'umm', 'like', 'think','right', 'duh', 'sigh', 'wheres', 'hmm', 'interesting', 'article','good','know',
        'say', 'hello', 'yup','im', 'ltsarcasmgt', 'hehe', 'blah', 'nope', 'ouch', 'uh']+stopwords.words('english')+get_stop_words('en'))
    
    doc_clean = []
    
    for doc in X.text:
        exclude = set(string.punctuation)
        lemma = WordNetLemmatizer()

        # Remove stop words
        stop_free = " ".join([i for i in doc.lower().split() if i not in set(stop)])
        
        # Remove '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
        punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
        
        # Lemmatize, normalize and replace certain words
        normalized = " ".join(re.sub('(hahaha|haha|lol)','funny',lemma.lemmatize(word.replace('doesnt',"doesn't").replace('dont',"don't"))) for word in punc_free.split())

        doc_clean.append(normalized)

    X['text'] = doc_clean
    return X

def CountVec(text_column):
	''' 
	Function to create a count vectorizer dataframe
	INPUT: Text column of dataframe (df_clean.text) 
	PARAMETERS: word appears in at least six documents, word appears in less than 50% of documents,
	only words with at least three letters
	OUTPUT: Count vectorizer, Count vectorizer dataframe
	'''

	cv = CountVectorizer(stop_words='english',min_df=6, max_df=0.5,token_pattern='\\b[A-Za-z][A-Za-z][A-Za-z]+\\b')
	cv_vecs = cv.fit_transform(text_column.transpose())

	return cv, cv_vecs

def TfidfVec(text_column):
	''' 
	Function to create a TFIDF vectorizer dataframe, head output: yes/no
	INPUT: Text column of dataframe
	PARAMETERS: word appears in at least six documents, word appears in less than 50% of documents,
	only words with at least three letters
	OUTPUT: TFIDF vectorizer dataframe
	'''

	vectorizer = TfidfVectorizer(stop_words='english',min_df=6, max_df=0.5,token_pattern='\\b[A-Za-z][A-Za-z][A-Za-z]+\\b')
	trans_vecs = vectorizer.fit_transform(text_column.transpose())
	trans_vecs_dense = pd.DataFrame(trans_vecs.todense(), columns=[vectorizer.get_feature_names()])
	
	return trans_vecs_dense

def gensim_corpus(vectorizer, vectorizer_df):
	''' 
	Export TFIDF vectors to gensim and let it know the mapping of row index to term:
		- Convert sparse matrix of counts to a gensim corpus
		- Transpose it for gensim -> needs to be terms by docs instead of docs by terms
	INPUT: Vectorizer, Vectorizer dataframe
	OUTPUT: Dictionary with id mapped to word, Corpus
	'''

	tfidf_corpus = matutils.Sparse2Corpus(vectorizer_df.transpose())
	# Row indices
	id2word = dict((v, k) for k, v in vectorizer.vocabulary_.items())
	id2word = corpora.Dictionary.from_corpus(tfidf_corpus, 
	                                         id2word=id2word)

	return id2word, tfidf_corpus

def LSI_space(mapped_dictionary, corpus, number_of_topics, words_per_topic):
	''' 
	Build an LSI space from the input vectorizer matrix, mapping of row id to word, and num_topics.
		- num_topics is the number of dimensions to reduce to after the SVD.
		- "fit" in sklearn, it primes an LSI space.
	INPUT: Id to word dictionary, gensim corpus, number of topics, words per topic
	OUTPUT: Topics
	'''

	lsi = models.LsiModel(corpus, id2word=mapped_dictionary, num_topics=number_of_topics)

	return lsi.show_topics(num_topics=number_of_topics, num_words=words_per_topic, formatted=False)



def LDA(mapped_dictionary, corpus, number_of_topics, words_per_topic):
	''' 
	Build a Latent Dirichlet Allocation (LDA) model
	INPUT: Id to word dictionary, gensim corpus, number of topics, words per topic
	OUTPUT: Topics
	'''
	
	lda = models.LdaModel(corpus, id2word=mapped_dictionary, num_topics=number_of_topics, passes=1,alpha = 0.01)
	
	return lda, lda.show_topics(number_of_topics,words_per_topic)


def run_model(saved_df):
	'''
	INPUT: Dataframe with user comments
	OUTPUT: Dataframe with most popular topics, Dataframe with individual user topics
	'''
	df = load_dataframe(saved_df)
	df_clean = clean_text(df)
	count_vectorizer, cv_vectors = CountVec(df_clean.text)
	id2word, vec_corpus = gensim_corpus(count_vectorizer, cv_vectors)
	lda_model, topics = LDA(id2word, vec_corpus, 30, 50)
	print(topics)

	# Save dataframe with topics and most frequent words
	topics_df = pd.DataFrame()
	for i in range(number_of_topics):
    	topics_df[i] = topics[i][1].split('+')

    topics_df.to_csv('LDA_topics.csv', index=False)

	# Retrieve vectors for the original tfidf corpus in the LSI space ("transform" in sklearn).
	# Dump the resulting document vectors into a list so we can take a look.
	lda_corpus = lda_model[corpus]
	doc_vecs = [doc for doc in lda_corpus]
	
	# Pick topic with highest proportion of all relevant topics for that comment.
	# Create columns with assigned topics to each comment.
	topics_per_user = [str(max(doc_vecs[i],key=operator.itemgetter(1))[0]) for i in range(len(df_clean))]
	df_clean['topics'] = topics_per_user
	
	# Concat topics for each user
	users_df = df_clean.loc[:,['author','topics']]
	users_df['topics'] = users_df.groupby(['author'])['topics'].transform(lambda x: ','.join(x))
	users_df = users_df.drop_duplicates()

	# Save dataframe with users and their top topics
	users_df.to_csv('users_df.csv',index=False)

run_model('all_comments.csv')



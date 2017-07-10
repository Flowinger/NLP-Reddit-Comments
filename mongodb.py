from __future__ import print_function
import requests

import pandas as pd
import numpy as np
from pymongo import MongoClient

def mongodb_comments(mongodb_ip):
	''' Connect to MongoDB database and get Reddit comments
	INPUT: IP to MongoDB database
	OUTPUT: Reddit comments
	'''
	client = MongoClient(mongodb_ip)
	db = client.redditdata
	comments = db.comments
	# Total number of comments in database
	print(comments.count())
	# Show first comment
	print(comments.find()[0])

	return comments

def get_comments(table):
	all_comments = []
	for obj in table.find():

	    if obj['body'] != '[deleted]':
	        all_comments.append({'author':obj['author'],'text':obj['body'],'controversiality':obj['controversiality'],
	                  'score':obj['score'], 'subreddit':obj['subreddit_id']})

	return all_comments

def extract():
	comments = mongodb_comments("mongodb://flow:pass@54.67.100.65/redditdata")
	df = get_comments(comments)

	# Save list with comments to a csv file
	pd.DataFrame(df).to_csv('all_comments.csv', index=False)

extract()

import csv
import numpy as np
from sklearn.svm import SVR #Support Vector Regression
import matplotlib.pyplot as plt
import tweepy
from textblob import TextBlob
from config import consumer_key,consumer_secret,access_token,access_secret

#configure Twitter Access
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)

api = tweepy.API(auth)

# Search for tweets that contain the word Machine Learning
public_tweets = api.search('Apple',result_type='popular')

for tweet in public_tweets:
    analysis = TextBlob(tweet.text) #Sentiment Analysis of the tweets
    if analysis.polarity > 0: #Assing a sentiment of Positive to every tweet which has polarity > 0
        sent = 'Positive'
    else:
        sent = 'Negative'
        

# Data Collection

dates = []
prices = []

def get_data(filename):
    with open(filename, 'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader) # to avoid indexes in the dataset
        for row in csvFileReader:
            dates.append(int(row[0].split('-')[2]))
            prices.append(float(row[1]))
    return

#reference csv file

get_data('AAPL.csv')

#Step 6 In this function, build your neural network model using Keras, train it, then have it predict the price 
#on a given day. We'll later print the price out to terminal.
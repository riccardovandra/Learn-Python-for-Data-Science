import  tweepy #import tweepy to interface Python with the Twitter API
from config import consumer_key,consumer_secret,access_secret,access_token
from textblob import TextBlob #  import textblog, a library for processing textual data

#configure Twitter Access
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)

api = tweepy.API(auth)

# Search for tweets that contain the word Trump
public_tweets = api.search('Trump')


for  tweet in public_tweets:
    print(tweet.text)
    analysis = TextBlob(tweet.text) #Do a sentiment analysis using textblob
    print(analysis.sentiment)

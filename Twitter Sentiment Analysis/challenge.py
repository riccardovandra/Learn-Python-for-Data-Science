"""

 save each Tweet to a CSV file with an associated label. The label should be either 'Positive' or 'Negative'. You can define the sentiment polarity threshold yourself, whatever you think constitutes a tweet being positive/negative.

 """

import tweepy
import csv
from textblob import TextBlob
from config import consumer_key,consumer_secret,access_token,access_secret

#configure Twitter Access
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)

api = tweepy.API(auth)

# Search for tweets that contain the word Machine Learning
public_tweets = api.search('Machine Learning',result_type='popular')

# Create CSV Data Initial List
csvData = [['Tweet','Sentiment(Number)','Sentiment','Objectivity(number)']]

for tweet in public_tweets:
    analysis = TextBlob(tweet.text) #Sentiment Analysis of the tweets
    if analysis.polarity > 0: #Assing a sentiment of Positive to every tweet which has polarity > 0
        sent = 'Positive'
    else:
        sent = 'Negative'
    csvData.append([tweet.text,analysis.polarity,sent,analysis.subjectivity]) #Append all the data in CSV List


#Write on CSV File
with open('ML_sentiment_Analysis.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(csvData) #Iterate trough the rows and write
csvFile.close()


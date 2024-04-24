import tweepy
from tweet import tweet
import authentication
from transformers import pipeline
import pandas as pd

# For use with the paid API
"""""
# Declare the API connection and the classifier
client = authentication.get_x_conn_v2()
classifier = pipeline('text-classifcation', model='src/models/FN_Truth_Seeker_Model/checkpoint-6710')

# Search term for finding tweets
query = "president -is:retweet"

# Get the tweets
response = client.search_recent_tweets(query, max_results=10)
# For each tweet, classify it and reply to it with the prediction result
for i in response.data:
    pred = classifier(i.text)
    reply = client.create_tweet(in_reply_to_tweet_id=i.id, text="Fake News Checker bot suggests this post is: " + pred[i]['label'] + " with a confidence of " + str(pred[i]['score']))

# test tweet with image
#tweet("API v2 with image test", "src/dog.jpg")
"""""

# For use with local data
# Load the data and the classifier
data = pd.read_csv("src/tweet_content.csv", on_bad_lines='skip')
classifier = pipeline('text-classification', model='src/models/FN_Truth_Seeker_Model/checkpoint-6710')

# Randomly sample 5 entries from the dataset
sample = data.sample(n=5)
print(sample)

# get predictions for each tweet in the sample
for index, row in sample.iterrows():
    text = row['tweet_text']
    pred = classifier(text)
    # Should limit the reply length to the tweet character limit
    print(f"Fake News Checker bot suggests this post is: {pred[0]['label']} with a confidence of {pred[0]['score'] * 100:.2f}%")
    tweet(f"In response to: {text}, Fake News Checker bot suggests this post is: {pred[0]['label']} with a confidence of {pred[0]['score'] * 100:.2f}%")


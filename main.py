from tweet import tweet
import authentication
from transformers import pipeline

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
    reply = client.create_tweet(in_reply_to_tweet_id=i.id, text="Fake News Checker bot suggests this post is: " + str(pred[i]['label']) + " with a confidence of " + str(pred[i]['score']))

# test tweet with image
#tweet("API v2 with image test", "src/dog.jpg")
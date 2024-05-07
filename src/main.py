import tweepy
from tweet import tweet
import authentication
from transformers import pipeline
import pandas as pd
import logging
import time

# Set up logging
logging.basicConfig(filename='bot_logs.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the model for predictions
classifier = pipeline('text-classification', model='src/models/FN_Truth_Seeker_Model/checkpoint-6710') # DistilBERT
#classifier = pipeline('text-classification', model='src/models/FN_TS_MobileBERT/checkpoint-26840')
#classifier = pipeline('text-classification', model='src/models/FN_TS_BertBaseUncased/checkpoint-26840')

"""""
#---------- For use with the paid API ----------#
def predict_and_tweet():
    # Connect to the API
    client = authentication.get_x_conn_v2()

    # Search term for finding tweets and the number of results to return
    query = "president -is:retweet"
    response = client.search_recent_tweets(query, max_results=10)

    # For each tweet, classify it and reply to it with the prediction result
    for i in response.data:
        pred = classifier(i.text)

        # Truncate the text if it is too long
        if len(i.text) > 180:
            i.text = i.text[:180]

        try:
            message = f"In response to: {i.text}, Fake News Checker bot suggests this post is: {pred[0]['label']} with a confidence of {pred[0]['score'] * 100:.2f}%"
            logging.info(message)
            # Reply to the tweet
            client.create_tweet(in_reply_to_tweet_id=i.id, text=message)
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            print(error_message)
            logging.error(error_message)

# Run the bot at a set interval
while True:
    predict_and_tweet()
    time.sleep(600) # 10 minutes
"""""

#---------- For use with local data and the free API ----------#
data = pd.read_csv("src/tweet_content.csv", on_bad_lines='skip')

def predict_and_test(state):
    # Randomly sample 5 entries from the dataset
    sample = data.sample(n=5, random_state=state)
    for _, row in sample.iterrows():
        text = row['tweet_text']
        pred = classifier(text)

        # Truncate the text if it is too long
        if len(text) > 180:
            text = text[:180]

        try:
            message = f"In response to: {text}, Fake News Checker bot suggests this post is: {pred[0]['label']} with a confidence of {pred[0]['score'] * 100:.2f}%"
            logging.info(message)
            print(message)
            tweet(message)
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            print(error_message)
            logging.error(error_message)

# Run the bot at a set interval
while True:
    state_counter = 1
    predict_and_test(state_counter)
    state_counter += 1
    time.sleep(600) # 10 minutes

import tweepy
from src import keys

# Authentication for APIv1, used for uploading images and videos
def get_x_conn_v1() -> tweepy.API:
    auth = tweepy.OAuthHandler(keys.API_KEY, keys.API_SECRET)
    auth.set_access_token(keys.ACCESS_TOKEN, keys.ACCESS_TOKEN_SECRET)
    
    return tweepy.API(auth)

# Authentication for APIv2, used for tweets
def get_x_conn_v2() -> tweepy.Client:
    client = tweepy.Client(
        consumer_key=keys.API_KEY,
        consumer_secret=keys.API_SECRET,
        access_token=keys.ACCESS_TOKEN,
        access_token_secret=keys.ACCESS_TOKEN_SECRET
    )
    
    return client
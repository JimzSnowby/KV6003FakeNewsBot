import authentication

client_v1 = authentication.get_x_conn_v1()
client_v2 = authentication.get_x_conn_v2()

def tweet(message: str, filename = None):
    try:
        if filename:
            media_id = client_v1.media_upload(filename).media_id_string
            client_v2.create_tweet(text=message, media_ids=[media_id])
            print("Tweeted with pic successfully!")
        else:
            client_v2.create_tweet(text=message)
            print("Tweeted successfully!")
    except Exception as e:
        print(e)
        print("Failed to tweet")
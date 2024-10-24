import tweepy

client = tweepy.Client("AAAAAAAAAAAAAAAAAAAAAAwWwgEAAAAAEx1MNFvrRnz07gZkUNaEQU0jyIg%3DtwiKDKTol20Fpe1D4sNdXVwtLcK87IFhlIqHvKsT5rmPvcCP26")

tweets = client.search_recent_tweets(query="generative ai", max_results=3)

print(tweets)
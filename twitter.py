# def get_tweets(topic, n=2):
#     url = "https://api.twitter.com/2/tweets/search/recent"
#     bearer_token = os.getenv("BEARER_TOKEN")
#     headers = {'Authorization': f'Bearer {bearer_token}', 'User-Agent': 'v2RecentSearchPython'}
#     query_params = {'query': '(from:twitterdev -is:retweet) OR #twitterdev','tweet.fields': 'author_id'}
#     response = requests.get(url, headers=headers, params=query_params)
#     print(response.status_code)
#     if response.status_code != 200:
#         raise Exception(response.status_code, response.text)
#     print(response.json())
#     return []

# def get_tweets(topic, n):
#     bearer_token = os.getenv("BEARER_TOKEN") 
#     # client = tweepy.Client(oauth2_bearer_token=os.environ.get("BEARER_TOKEN"))
#     oauth2_user_handler = tweepy.OAuth2UserHandler(
#         client_id="bnlyek5oc1lNd2V4VzJpZlNNQ2g6MTpjaQ",
#         redirect_uri="https://www.nowebsite.com/",
#         scope=["tweet.read"],
#         # Client Secret is only necessary if using a confidential client
#         client_secret="gZsMysm-LK9h27H_4oqfzjZ6VZG9fcXfNj42JTgk4jZUJSiD_C"
#     )
#     print(oauth2_user_handler.get_authorization_url())
#     access_token = oauth2_user_handler.fetch_token(
#         "https://www.nowebsite.com/"
#     )
#     client = tweepy.Client(access_token)
#     tweets = client.search_recent_tweets(query=topic, max_results=n)
#     return tweets
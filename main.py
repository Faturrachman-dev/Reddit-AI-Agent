from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, ChatMessagePromptTemplate
# from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.schema.document import Document
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from dotenv import load_dotenv
from praw import Reddit
import requests
import os
import json
import tweepy
import json

load_dotenv()

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

def get_reddits(topic,n):
    reddit = Reddit(
        client_id=os.environ.get("REDDIT_API_CLIENT_ID"),
        client_secret=os.environ.get("REDDIT_API_SECRET"),
        user_agent=os.environ.get("REDDIT_USER_AGENT"),
    )

    res = []
    for submission in reddit.subreddit("all").search(query=topic, sort="relevance", limit=n):
        print(submission.title)
        res.append(submission)

    comments = []
    for value in res:
        res=""
        submission = reddit.submission(id=value)
        res = submission.title + "\n"
        submission.comments.replace_more(limit=0)
        for comment in submission.comments.list():
            res+=comment.body
        # loader = TextLoader(res, encoding='utf-8')
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = [Document(page_content=x) for x in text_splitter.split_text(res)]
        comments.extend(docs)

    return comments
    

def summarize_tweets(documents):

    llm = ChatGroq(model="gemma2-9b-it")

    prompt = ChatPromptTemplate.from_messages([
        ('system',"You are provided with twitter thread comments. Most of the comments are discussions related to problems and solutions users are facing. Creat a concise summary of all the solutions and suggestions and key points mentioned in the discussions. \\n ===Comments=== \\n {data}")
    ])

    # map_prompt_template = """You are provided with twitter thread comments. Most of the comments are discussions related to problems and solutions users are facing. Creat a concise summary of all the solutions and suggestions and key points mentioned in the discussions. \\n ===Comments=== \\n {text}"""

    # map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])

    # combine_prompt_template = """You are provided with twitter thread comments. Most of the comments are discussions related to problems and solutions users are facing. Creat a concise summary of all the solutions and suggestions and key points mentioned in the discussions. \\n ===Comments=== \\n {text}"""

    # combine_prompt = PromptTemplate(
    #     template=combine_prompt_template, input_variables=["text"]
    # )

    # map_reduce_chain = load_summarize_chain(
    #     llm,
    #     chain_type="map_reduce",
    #     map_prompt=map_prompt,
    #     combine_prompt=combine_prompt,
    #     return_intermediate_steps=True,
    # )

    res = []
    # try:
    # text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    #     chunk_size=1000, chunk_overlap=100
    # )
    
    chain = prompt | llm
    # split_docs = []
    # for doc in documents:
    #     new_docs = text_splitter.split_documents(doc)
    #     split_docs.append(new_docs)
    #     print(f"Generated {len(split_docs)} documents.")
    
    for doc in documents:
        result = chain.invoke({'data':doc})
        res.append(result.content)

    # except Exception as e:
    #     print('Exception Occured')
    #     print(e)
    #     pass
        
    return res

def main():
    topic = input("Please enter topic you want to know about!! ")
    n = int(input("Enter no of tweets you want to capture "))

    # Get the tweets and save it in a file for recursive use
    documents = get_reddits(topic,n)
    # with open("documents_safe.txt", "w", encoding="utf-8") as documents_file_safe:
    #     json.dump(documents, documents_file_safe)
    # with open("documents_safe.txt", "r", encoding="utf-8") as documents_file_safe:
    #     documents = json.load(documents_file_safe)

    summary = summarize_tweets(documents)
    # for data in summary:
    #     print(data)
    #     print('-'*20)
    # print('='*20)
    # for data in documents:
    #     print(data)
    #     print('-'*20)
    
    # Save summary to a file named "summary.txt"
    with open("summary.txt", "w", encoding="utf-8") as summary_file:
        # for doc in summary:
        #     summary_file.write(doc)
        #     summary_file.write("\n")
        json.dump(summary, summary_file)
        # for data in summary:
        #     summary_file.write(data + "\n")
        #     summary_file.write("-" * 20 + "\n")

    # Save documents to a file named "documents.txt"
    # with open("documents.txt", "w", encoding="utf-8") as documents_file:
    #     for doc in documents:
    #         summary_file.write(doc)
    #         summary_file.write("\n")
        # json.dump(documents, documents_file)
        # for data in documents:
        #     documents_file.write(data + "\n")
        #     documents_file.write("-" * 20 + "\n")

    print("Summary and documents saved to files.")

if __name__ == "__main__":
    main()
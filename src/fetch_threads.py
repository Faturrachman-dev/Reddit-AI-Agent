from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, ChatMessagePromptTemplate
# from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
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

# RAG
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage

def get_reddit(topic,n):
    reddit = Reddit(
        client_id=os.environ.get("REDDIT_API_CLIENT_ID"),
        client_secret=os.environ.get("REDDIT_API_SECRET"),
        user_agent=os.environ.get("REDDIT_USER_AGENT"),
    )

    tweets = []
    print("Reddit Threads Title: ")
    for submission in reddit.subreddit("all").search(query=topic, sort="relevance", limit=n):
        tweets.append(submission)

    comments = []
    data = ''
    details = ''
    idx = 1
    for value in tweets:
        submission = reddit.submission(id=value)
        res = submission.title + "\n"
        submission.comments.replace_more(limit=0)
        # Logging
        print(f"{idx}. {submission.title} - {len(submission.comments)} comments")
        details+=f"{idx}. {submission.title} - {len(submission.comments)} comments" + "\n"
        # Extract All level comments using .list() function -> we can extract upro 8000 comments but a lot of it is irrevelant Data
        # for comment in submission.comments.list():
        #     res+=comment.body

        # Only extracting first level comment so we have most important data
        for top_level_comment in submission.comments:
            res+=top_level_comment.body
        comments.append(submission.title+'$$'+res)
        data+=res
        idx+=1

    details+=f"Total Token: {len(data.split())}"

    # Save tweet data in a file
    with open("tweet_data.txt", "w", encoding="utf-8") as tweet_data:
        json.dump(comments, tweet_data)
    
    return details
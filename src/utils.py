
from praw import Reddit
import os

reddit = Reddit(
    client_id=os.environ.get("REDDIT_API_CLIENT_ID"),
    client_secret=os.environ.get("REDDIT_API_SECRET"),
    user_agent=os.environ.get("REDDIT_USER_AGENT"),
)

# Change you models here
from langchain_groq import ChatGroq
llm = ChatGroq(model="llama3-8b-8192", temperature=0)
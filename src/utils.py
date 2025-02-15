from praw import Reddit
import os
from dotenv import load_dotenv

load_dotenv()

reddit = Reddit(
    client_id=os.environ.get("REDDIT_API_CLIENT_ID"),
    client_secret=os.environ.get("REDDIT_API_SECRET"),
    user_agent=os.environ.get("REDDIT_USER_AGENT"),
)

# Change you models here
# from langchain_groq import ChatGroq # Remove this
# llm = ChatGroq(model="llama3-70b-8192", temperature=0) # Remove this

# NEW:  No need for llm here anymore; it's in combined_CLI.py and UI.py
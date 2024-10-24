import os
from praw import Reddit
from praw.models import MoreComments
from dotenv import load_dotenv
load_dotenv()

reddit = Reddit(
        client_id=os.environ.get("REDDIT_API_CLIENT_ID"),
        client_secret=os.environ.get("REDDIT_API_SECRET"),
        user_agent=os.environ.get("REDDIT_USER_AGENT"),
    )

res = []
for submission in reddit.subreddit("all").search(query="scrape twitter tweets using python", sort="relevance", limit=3):
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
        print(comment.body)
    comments.append(res)

print(len(res))
print(res)
print('\n\n')
print(len(comments))
print(comments)
from utils import reddit
import json

def get_reddit(topic,n):
    tweets = []
    print("\nReddit Threads Title: ")
    for submission in reddit.subreddit("all").search(query=topic, sort="relevance", limit=n):
        tweets.append(submission)

    comments = []
    combined = ''
    reddit_threads_data = ''
    for idx, value in enumerate(tweets):
        submission = reddit.submission(id=value)
        res = submission.title + "\n"
        submission.comments.replace_more(limit=0)
        # Logging
        print(f"{idx+1}. {submission.title} - {len(submission.comments)} comments")
        reddit_threads_data+=f"{idx+1}. {submission.title} - {len(submission.comments)} comments" + "\n"
        # Extract All level comments using .list() function -> we can extract upto 8000 comments but a lot of it is irrevelant combined
        # for comment in submission.comments.list():
        #     res+=comment.body

        # Only extracting first level comment so we have most important Data
        for top_level_comment in submission.comments:
            res+=top_level_comment.body
        comments.append(submission.title+'$$'+res)
        combined+=res

    reddit_threads_data+=f"Total Token: {len(combined.split())}"

    # Save tweet data in a file
    with open("../data/reddit_threads_data.txt", "w", encoding="utf-8") as tweet_data:
        json.dump(comments, tweet_data)
    
    return reddit_threads_data
import praw
import os
from dotenv import load_dotenv

load_dotenv()

reddit = praw.Reddit(
    client_id=os.getenv('REDDIT_CLIENT_ID'),
    client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
    user_agent='Comment extraction'
)


subreddit = reddit.subreddit('worldnews')
posts = subreddit.top('day', limit=1000)
for post in posts:
    print(post.title)
    print(post.score)
    print(post.num_comments)
    print(post.selftext)
    print(post.id)
    print(post.url)
    permalink = post.permalink
    submission = reddit.submission(url=f'https://www.reddit.com{permalink}')
    submission.comment_sort = 'best'
    submission.comment_limit = 5
    for top_level_comment in submission.comments:
        print(top_level_comment.body)
    break
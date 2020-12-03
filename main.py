import praw
from praw.models import MoreComments
import os
from dotenv import load_dotenv
import pickle
load_dotenv()

class RedditCrawler:
    def __init__(self):
        self.reddit = praw.Reddit(
            client_id=os.getenv('REDDIT_CLIENT_ID'),
            client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
            user_agent='Comment extraction'
        )

    def fetchAndSaveTopPostsOfToday(self):
        top_subreddits = []
        top_comments = []

        subreddit = self.reddit.subreddit('worldnews')
        posts = subreddit.top('month', limit=1000)
        for i, post in enumerate(posts):
            postData = {
                'title': post.title,
                'score': post.score,
                'num_comments': post.num_comments,
                'body': post.selftext,
                'id': post.id,
                'url': post.url,
                'permalink': post.permalink,
                'created_utc': post.created_utc
            }
            top_subreddits.append(postData)

            submission = self.reddit.submission(url=f'https://www.reddit.com{postData["permalink"]}')
            submission.comment_sort = 'best'
            submission.comment_limit = 100
            for comment in submission.comments:
                # check if end of page
                if isinstance(comment, MoreComments):
                    continue
                commentData = {
                    'id': comment.id,
                    'body': comment.body,
                    'author': comment.author,
                    'score': comment.score,
                    'created_utc': comment.created_utc,
                    'postId': postData['id']
                }
                top_comments.append(commentData)

            if i % 10 == 0:
                print(f'Crawled {i}/1000')

        with open('top_subreddits.pkl', 'wb') as f:
            pickle.dump(top_subreddits, f)
            f.close()

        with open('top_comments.pkl', 'wb') as f:
            pickle.dump(top_comments, f)
            f.close()

if __name__ == "__main__":
    redditCrawler = RedditCrawler()
    redditCrawler.fetchAndSaveTopPostsOfToday()
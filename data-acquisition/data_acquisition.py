import praw
from praw.models import MoreComments
import os
from dotenv import load_dotenv
from datetime import date
import pickle
load_dotenv()

class RedditCrawler:
    def __init__(self):
        self.reddit = praw.Reddit(
            client_id=os.getenv('REDDIT_CLIENT_ID'),
            client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
            user_agent='Comment extraction'
        )

    def fetchAndSaveTopPosts(self, subreddit, timeframe):
        if timeframe == 'all':
            post_limit = 10000
            comment_limit = 200
        elif timeframe == 'year':
            post_limit = 10000
            comment_limit = 100
        elif timeframe == 'day':
            post_limit = 50
            comment_limit = 10
        else:
            print('Not configured...')
            exit()

        top_subreddits = []
        top_comments = []
        today = date.today().isoformat()

        subreddit = self.reddit.subreddit(subreddit)
        posts = subreddit.top(timeframe, limit=post_limit)
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
            submission.comment_limit = comment_limit
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
                print(f'{subreddit}: Crawled {i} posts')

        with open(f'./data/{today}_{timeframe}_{subreddit}_subreddits.pkl', 'wb') as f:
            pickle.dump(top_subreddits, f)
            f.close()

        with open(f'./data/{today}_{timeframe}_{subreddit}_comments.pkl', 'wb') as f:
            pickle.dump(top_comments, f)
            f.close()

if __name__ == "__main__":
    print('Starting reddit fetch script...')
    subreddits = ['worldnews', 'news', 'politics', 'upliftingnews', 'truenews']
    redditCrawler = RedditCrawler()
    
    for subreddit in subreddits:
        print(f'Fetch data from {subreddit}')
        redditCrawler.fetchAndSaveTopPosts(subreddit, 'year')
    
    print('Finish!')
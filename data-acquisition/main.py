"""
Script to crawl Reddit data using PRAW and pushshift
"""

import requests
import praw
from praw.models import MoreComments
import os
from dotenv import load_dotenv
from datetime import date
import pickle
load_dotenv()

def load_data (subreddit,days):
    """grabs 10 submissions per day for the given amount of days before 20.01.2021"""
    i = 0
    after = 1611010800
    before = 1611097199
    submission_ids=[]
    while i < days:
        try:
            page_data = requests.get(f'http://api.pushshift.io/reddit/search/submission/?subreddit={subreddit}&size=10&before={before}&after={after}&sort=desc&sort_type=score').json()
        except:
            continue
        #print(page_data['data'])
        for submission in page_data['data']:
            submission_ids.append(submission['id'])
        after -= 86400
        before -= 86400
        i+=1
    return submission_ids 
    
def load_fetch_data (subreddit, days, saving_timeframe=30):
    """grabs 10 submissions per day for the given amount of days before 20.01.2021 and saves the top 10 comments each"""
    i = 0
    after = 1611010800
    before = 1611097199
    #timeframe is ... days
    timeframe = saving_timeframe
    tf_begin = 0
    tf_end = 0
    tf_posts = []
    tf_comments = []
    total_posts = 0
    total_comments = 0

    for i in range(days):
        #setup file name
        if i%timeframe == 0:
            tf_end = before
        
        #get submission ids for the day
        submission_ids = []
        top_posts = []
        try:
            page_data = requests.get(f'http://api.pushshift.io/reddit/search/submission/?subreddit={subreddit}&size=10&before={before}&after={after}&sort=desc&sort_type=score').json()
        except:
            continue
        for submission in page_data['data']:
            submission_ids.append(submission['id'])
            postData = {
                    'title': submission['title'],
                    'score': submission['score'],
                    #'num_comments': submission['num_comments'],
                    #'body': submission['selftext'],
                    'id': submission['id'],
                    #'url': submission['url'],
                    #'permalink': submission['permalink'],
                    #'created_utc': submission['created_utc']
                }
            top_posts.append(postData)
        #get top post/comment information
        top_comments_all = fetch_comments(submission_ids)
        tf_posts.extend(top_posts)
        tf_comments.extend(top_comments_all)

        #write data to file after [timeframe] days
        if i%timeframe == timeframe - 1:
            tf_begin = after
            with open(f'./data/{tf_begin}_to_{tf_end}_{subreddit}_posts.pkl', 'wb') as f:
                pickle.dump(tf_posts, f)
                f.close()
            with open(f'./data/{tf_begin}_to_{tf_end}_{subreddit}_comments.pkl', 'wb') as f:
                pickle.dump(tf_comments, f)
                f.close()
            total_posts += len(tf_posts)
            total_comments += len(tf_comments)
            tf_posts = []
            tf_comments = []
            print(f'{subreddit}: Crawled {total_posts} posts and got {total_comments} comments. {i}/{days}')
        
        if i%10 == 0:
            print(f'{i}/{days}')

        #prepare next step
        after -= 86400
        before -= 86400
    
    #clean up leftovers (write more data)
    if len(tf_posts) != 0:
        tf_begin = after + 86400
        with open(f'./data/{tf_begin}_to_{tf_end}_{subreddit}_posts.pkl', 'wb') as f:
            pickle.dump(tf_posts, f)
            f.close()
        with open(f'./data/{tf_begin}_to_{tf_end}_{subreddit}_comments.pkl', 'wb') as f:
            pickle.dump(tf_comments, f)
            f.close()
        total_posts += len(tf_posts)
        total_comments += len(tf_comments)
    
    print(f'{subreddit}: Crawled {total_posts} posts and got {total_comments} comments.')
    return total_posts,total_comments

def fetch_comments (ids):
    """grabs comments via praw for set of submission ids"""
    top_comments_all = []

    #iterate over postids
    for post_id in ids:
        post = reddit.submission(id=post_id)

        #format praw post comments
        post.comments.replace_more(limit=1)
        post.comment_sort = 'top'
        post.comment_limit = 20
        top_comments = []

        for i,comment in enumerate(post.comments):
            # check if end of page
            if isinstance(comment, MoreComments):
                print(i)
                continue
            #save post comment information to list
            commentData = {
                'id': comment.id,
                'body': comment.body,
                #'author': comment.author,
                'score': comment.score,
                #'created_utc': comment.created_utc,
                'postId': post_id
            }
            top_comments.append(commentData)

        #filter 10 best comments
        top_comments.sort(key=lambda comment: comment["score"], reverse=True)
        top_comments_all.extend(top_comments[:20])

    return top_comments_all
    

if __name__ == "__main__":
    
    subreddits = ['worldnews', 'news', 'politics', 'upliftingnews', 'truenews']
    reddit = praw.Reddit(client_id=os.getenv('REDDIT_CLIENT_ID'),client_secret=os.getenv('REDDIT_CLIENT_SECRET'),user_agent='Comment extraction')
    num_posts = 0
    num_comments = 0
    
    for subreddit in subreddits:
        print(f'Fetching data from {subreddit}')
        posts,comments = load_fetch_data(subreddit,1000,180)
        num_posts += posts
        num_comments += comments

    print(f'GRAND TOTAL: Crawled {num_posts} posts and got {num_comments} comments in {len(subreddits)} subreddits.')

    print('Done!')

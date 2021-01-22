import requests
import praw
from praw.models import MoreComments
import os
from dotenv import load_dotenv
from datetime import date
import pickle
load_dotenv()

def load_data (subreddit):
    i = 0
    after = 1611010800
    before = 1611097199
    submission_ids=[]
    while i < 10:
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
    
def fetch_posts (ids,comment_limit):
    top_subreddits = []
    top_comments = []


    all_comments = [] #REDDIT answer

    for post_id in ids:
        post = reddit.submission(id=post_id)
        #print(post)
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
        #print(postData)
        top_subreddits.append(postData)
        print(f'Post ID: {post_id}')

        post.comments.replace_more(limit=1)
        post.comment_sort = 'top'
        post.comment_limit = comment_limit
        
        #REDDIT ANSWER
        #submission_comments = praw.helpers.flatten_tree(post.comments)
        #don't include non comment objects such as "morecomments"
        #real_comments = [comment for comment in submission_comments if isinstance(comment, praw.objects.Comment)]
        #real_comments.sort(key=lambda comment: comment.score, reverse=True)
        #toppp_comments = real_comments[:25] #top 25 comments
        #print(f'amount of comments here: {len(toppp_comments)}')
        #print(toppp_comments)
        comments123 = []

        for comment in post.comments:
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
            print(f'Comment score: {commentData["score"]}')
            comments123.append(commentData)
        comments123.sort(key=lambda comment: comment["score"], reverse=True)
        for comment in comments123:
            print(f'123Comment score: {comment["score"]}')
    print(f'Crawled {len(top_subreddits)} posts.')
    print(f'Got {len(top_comments)} comments.')
    #print(top_comments)
    



if __name__ == "__main__":
    
    subreddits = ['worldnews', 'news', 'politics', 'upliftingnews', 'truenews']
    reddit = praw.Reddit(client_id=os.getenv('REDDIT_CLIENT_ID'),client_secret=os.getenv('REDDIT_CLIENT_SECRET'),user_agent='Comment extraction')
    ids = []
    #for subreddit in subreddits:
    #    print(f'Fetching data from {subreddit}')
    #    ids.extend(load_data(subreddit))
    #    print(len(ids))
    
    ids = ['l0tuef', 'l0szrj', 'k4qide']
    
    print(f'Got following submission IDs: {ids}')
    print(len(ids))
    x = set(ids)
    print(len(x))

    fetch_posts(ids,10)

    print('Done!')

    #y = requests.get('http://api.pushshift.io/reddit/search/submission/?subreddit=worldnews&size=10&before=1608312456&sort=asc')
    #print(y.text)
    #z = parse("It's {}, I love it!", "It's spam, I love it!")
    #print(z)
    #a = search('Age: {:d}', 'Name: Rufus\nAge: 42\nColor: red\nAge: 32')
    #print(a)
    #print(y.json()['data'][0]['id'])
    #b = y.text.split("},\n        {")
    #print(b[0])
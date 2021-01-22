import pickle
import os

from api.config import RAW_DATA_PATH, PROCESSED_DATA_PATH
from api.utils import preprocess_sentence

DIRNAME = os.path.dirname(__file__)

class Preprocessing():
    def __init__(self):
        print('Initialize preprocessing')

        self.filenames = os.listdir(os.path.join(DIRNAME, RAW_DATA_PATH))
        orig_subreddits, orig_comments = self._load_data()
        filtered_comments = self._filter_out_negatives(orig_comments)
        subreddit_comment_pair = self._rearrange_data(filtered_comments, orig_subreddits)
        for i, pair in enumerate(subreddit_comment_pair):
            subreddit_comment_pair[i]['comment'] = self._filter_by_character_and_pad(pair['comment'], '! ', True)
            subreddit_comment_pair[i]['comment'] = self._filter_by_character_and_pad(pair['comment'], '? ', True)
            subreddit_comment_pair[i]['comment'] = self._filter_by_character_and_pad(pair['comment'], '. ', True)
            subreddit_comment_pair[i]['comment'] = self._filter_by_character_and_pad(pair['comment'], '\n', False)
        self.headlines = []
        self.comments = []
        for pair in subreddit_comment_pair:
            self.headlines.append(preprocess_sentence(pair['headline']))
            self.comments.append(preprocess_sentence(pair['comment']))

    def _load_data(self):
        orig_subreddits = []
        orig_comments = []
        for filename in self.filenames:
            if 'subreddits' in filename:
                with open(os.path.join(DIRNAME, RAW_DATA_PATH, filename), 'rb') as f:
                    orig_subreddits.extend(pickle.load(f))
            if 'comments' in filename:
                with open(os.path.join(DIRNAME, RAW_DATA_PATH, filename), 'rb') as g:
                    orig_comments.extend(pickle.load(g))

        return orig_subreddits, orig_comments

    # filter out subreddits and comments with negative score
    def _filter_out_negatives(self, input):
        output = []
        for i in input:
            if i['score'] > 0:
                output.append(i)
        return output

    def _get_subreddit_title(self, subreddit_id, orig_subreddits):
        for subreddit in orig_subreddits:
            if subreddit['id'] == subreddit_id:
                return subreddit['title']
    
    def _rearrange_data(self, filtered_comments, orig_subreddits):
        subreddit_comment_pair = []
        for comment in filtered_comments:
            comment_body = comment['body']
            subreddit_id = comment['postId']
            subreddit_title = self._get_subreddit_title(subreddit_id, orig_subreddits)
            subreddit_comment_pair.append({
                "headline": subreddit_title,
                "comment": comment_body
            })
        return subreddit_comment_pair

    def _filter_by_character_and_pad(self, paragraph, sign, pad):
        new = paragraph.split(sign)[0]
        if new == paragraph or not pad:
            return new
        return new + sign


if __name__ == '__main__':
    preprocessing = Preprocessing()
    with open(os.path.join(DIRNAME, PROCESSED_DATA_PATH, f'processed_headlines.pkl'), 'wb') as f:
        pickle.dump(preprocessing.headlines, f)
        f.close()

    with open(os.path.join(DIRNAME, PROCESSED_DATA_PATH, f'processed_comments.pkl'), 'wb') as f:
        pickle.dump(preprocessing.comments, f)
        f.close()
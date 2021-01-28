import pickle
import os

from api.config import RAW_DATA_PATH, FILTER_WORDS, MODEL_PATH
from api.utils import preprocess_sentence

class Preprocessing():
    def __init__(self, preprocessed, selected_model='', save_preprocessed=False):
        print('Initialize preprocessing')
        if (preprocessed):
            print('Use preprocessed data...')
            self.model_path = os.path.join(MODEL_PATH, selected_model)
            self.headlines = self._load_processed_data(self.model_path, 'processed_headlines.txt')
            self.comments = self._load_processed_data(self.model_path, 'processed_comments.txt')
        else:
            self.filtered_count = 0
            orig_subreddits, orig_comments = self._load_data()
            filtered_comments = self._filter_out_keywords(orig_comments)
            filtered_comments = self._filter_out_negatives(filtered_comments)
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
            if save_preprocessed:
                self._save_to_txt('processed_headlines.txt', self.headlines)
                self._save_to_txt('processed_comments.txt', self.comments)

    def _load_processed_data(self, preprocessed_path, filename):
        file = open(os.path.join(preprocessed_path, filename), 'r')
        rows = file.readlines()
        data = []
        for row in rows:
            data.append(row.split('\n')[0])
        return data

    def _load_data(self):
        orig_subreddits = []
        orig_comments = []
        filenames = os.listdir(RAW_DATA_PATH)
        for filename in filenames:
            if 'posts' in filename:
                with open(os.path.join(RAW_DATA_PATH, filename), 'rb') as f:
                    orig_subreddits.extend(pickle.load(f))
            if 'comments' in filename:
                with open(os.path.join(RAW_DATA_PATH, filename), 'rb') as g:
                    orig_comments.extend(pickle.load(g))

        return orig_subreddits, orig_comments

    # filter out subreddits and comments with negative score
    def _filter_out_negatives(self, input):
        output = []
        for i in input:
            if i['score'] > 0:
                output.append(i)
            else:
                self.filtered_count += 1
        return output
    
    def _filter_out_keywords(self, input):
        output = []
        for i in input:
            for keyword in FILTER_WORDS:
                if keyword in i['body']:
                    self.filtered_count += 1
                    break
            else:
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

    def _save_to_txt(self, filename, data):
        file = open(os.path.join(self.model_path, filename), 'w')
        for d in data:
            file.write("%s\n" % d)


if __name__ == '__main__':
    preprocessing = Preprocessing(save_preprocessed=True)
    print(preprocessing.filtered_count)
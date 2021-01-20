import pickle
import re
import os
import unicodedata

class Preprocessing():
    def __init__(self):
        self.filenames = os.listdir('../../data')
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
            self.headlines.append(self.preprocess_sentence(pair['headline']))
            self.comments.append(self.preprocess_sentence(pair['comment']))

    def _load_data(self):
        orig_subreddits = []
        orig_comments = []
        for filename in self.filenames:
            if 'subreddits' in filename:
                with open(filename, 'rb') as f:
                    self.orig_subreddits.extend(pickle.load(f))
            if 'comments' in filename:
                with open(filename, 'rb') as g:
                    self.orig_comments.extend(pickle.load(g))

        return orig_subreddits, orig_comments

    # filter out subreddits and comments with negative score
    def _filter_out_negatives(self, input):
        output = []
        for i in input:
            if i['score'] > 0:
                output.append(i)
        return output

    def preprocess_sentence(self, w):
        w = w.lower().strip()
        # This next line is confusing!
        # We normalize unicode data, umlauts will be converted to normal letters
        w = w.replace("ß", "ss")
        w = ''.join(c for c in unicodedata.normalize('NFD', w) if unicodedata.category(c) != 'Mn')

        # creating a space between a word and the punctuation following it
        # eg: "he is a boy." => "he is a boy ."
        # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
        w = re.sub(r"([?.!,¿])", r" \1 ", w)
        w = re.sub(r'[" "]+', " ", w)

        # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
        w = re.sub(r"[^a-zA-Z?.!]+", " ", w)
        w = w.strip()

        # adding a start and an end token to the sentence
        # so that the model know when to start and stop predicting.
        w = '<start> ' + w + ' <end>'
        return w

    def _get_subreddit_title(self, subreddit_id):
        for subreddit in self.orig_comments:
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

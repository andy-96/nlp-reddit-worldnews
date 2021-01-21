import os
import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split

from src.config import PROCESSED_DATA_PATH, BATCH_SIZE

class Dataset():
    def __init__(self):
        # Load data
        with open(os.path.join(PROCESSED_DATA_PATH, 'processed_comments.pkl'), "rb") as f:
            self.comments = pickle.load(f)
        with open(os.path.join(PROCESSED_DATA_PATH, 'processed_headlines.pkl'), "rb") as f:
            self.headlines = pickle.load(f)
        
        # Initialize tokenizer
        self.max_length_input = self.max_len(self.comments)
        self.headline_tokenizer, self.comment_tokenizer = self.get_tokenized_headline()
        self.input_vocab_size = len(self.headline_tokenizer.word_index) + 1  
        self.target_vocab_size = len(self.comment_tokenizer.word_index) + 1

        self.buffer_size = 0
        self.dataset = self.create_dataset()
        self.steps_per_epoch = self.buffer_size // BATCH_SIZE

    def tokenize(self):
        headline_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
        headline_tokenizer.fit_on_texts(self.headlines)

        comment_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
        comment_tokenizer.fit_on_texts(self.comments)

    def create_dataset(self):
        data_headline = self.headline_tokenizer.texts_to_sequences(self.headlines)
        data_headline = tf.keras.preprocessing.sequence.pad_sequences(data_headline, padding='post')

        data_comment = self.comment_tokenizer.texts_to_sequences(self.comments)
        data_comment = tf.keras.preprocessing.sequence.pad_sequences(data_comment,padding='post')

        X_train,  X_test, Y_train, Y_test = train_test_split(data_headline, data_comment, test_size=0.2)
        self.buffer_size = len(X_train)

        return tf.data.Dataset.from_tensor_slices((X_train, Y_train)).shuffle(self.buffer_size).batch(BATCH_SIZE, 
                                                                                            drop_remainder=True) 
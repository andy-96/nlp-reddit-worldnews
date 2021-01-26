import tensorflow as tf
from sklearn.model_selection import train_test_split

from api.preprocessing import Preprocessing
from api.config import PROCESSED_DATA_PATH, BATCH_SIZE

class Dataset():
    def __init__(self):
        print('Initialize dataset')
        # Load data
        self.preprocessing = Preprocessing()
        
        # Initialize tokenizer
        self.max_length_output = 0
        self.headline_tokenizer, self.comment_tokenizer = self.tokenize()
        self.input_vocab_size = len(self.headline_tokenizer.word_index) + 1  
        self.target_vocab_size = len(self.comment_tokenizer.word_index) + 1

        self.buffer_size = 0
        self.dataset = self.create_dataset()
        self.steps_per_epoch = self.buffer_size // BATCH_SIZE

    def max_len(self, sentence):
        return max(len(s) for s in sentence)

    def tokenize(self):
        headline_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
        headline_tokenizer.fit_on_texts(self.preprocessing.headlines)

        comment_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
        comment_tokenizer.fit_on_texts(self.preprocessing.comments)

        return headline_tokenizer, comment_tokenizer

    def create_dataset(self):
        data_headline = self.headline_tokenizer.texts_to_sequences(self.preprocessing.headlines)
        data_headline = tf.keras.preprocessing.sequence.pad_sequences(data_headline, padding='post')

        data_comment = self.comment_tokenizer.texts_to_sequences(self.preprocessing.comments)
        data_comment = tf.keras.preprocessing.sequence.pad_sequences(data_comment,padding='post')
        self.max_length_output =  self.max_len(data_comment)

        X_train,  X_test, Y_train, Y_test = train_test_split(data_headline, data_comment, test_size=0.2)
        self.buffer_size = len(X_train)

        return tf.data.Dataset.from_tensor_slices((X_train, Y_train)).shuffle(self.buffer_size).batch(BATCH_SIZE, 
                                                                                            drop_remainder=True) 
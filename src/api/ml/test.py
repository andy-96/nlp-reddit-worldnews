from tensorflow.keras import preprocessing
import tensorflow as tf
import os
import pickle
from ..config import PROCESSED_DATA_PATH

class CommentGenerator():
    def __init__(self):
        with open(os.join.path(PROCESSED_DATA_PATH, 'processed_comments.pkl'), "rb") as f:
            self.comments = pickle.load(f)
        with open(os.join.path(PROCESSED_DATA_PATH, 'processed_headlines.pkl'), "rb") as f:
            self.headlines = pickle.load(f)
        
        self.max_length_input = self.max_len(self.comments)

    def max_len(self, sentence):
        return max(len(s) for s in sentence)

    def get_tokenized_headline(self):
        headline_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
        headline_tokenizer.fit_on_texts(self.headlines)

        comment_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
        comment_tokenizer.fit_on_texts(self.comments)

        return headline_tokenizer, comment_tokenizer


    def create_padding_mask(self, seq):
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
        
        # add extra dimensions to add the padding
        # to the attention logits. Tensor sizes are always a pain...
        return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


    def create_look_ahead_mask(self, size):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask


    def create_masks(self, input, target):
        # Encoder padding mask
        encoder_padding_mask = self.create_padding_mask(input)
        
        # Used in the 2nd attention block in the decoder.
        # This padding mask is used to mask the encoder outputs.
        decoder_padding_mask = self.create_padding_mask(input)
        
        # Used in the 1st attention block in the decoder.
        # It is used to pad and mask future tokens in the input received by 
        # the decoder.
        look_ahead_mask = self.create_look_ahead_mask(tf.shape(target)[1])
        decoder_target_padding_mask = self.create_padding_mask(target)
        combined_mask = tf.maximum(decoder_target_padding_mask, look_ahead_mask)
        
        return encoder_padding_mask, decoder_padding_mask, combined_mask


    def translate(self, sentence):
        headline_tokenizer, comment_tokenizer = self.get_tokenized_headline()

        sentence = preprocessing.preprocess_sentence(sentence)
        input = headline_tokenizer.texts_to_sequences([sentence])
        input = tf.convert_to_tensor(input)
        
        # as the target is German, the first word to the transformer should be the
        # German start token
        start_token = tf.convert_to_tensor([comment_tokenizer.word_index['<start>']])
        start_token = tf.expand_dims(start_token, 0)

        # And it should stop with the end token
        end_token = tf.convert_to_tensor(comment_tokenizer.word_index['<end>'])
        end_token = tf.expand_dims(end_token, 0)
        
        output = start_token
        for i in range(self.max_length_output):
            enc_padding_mask, dec_padding_mask, combined_mask, = self.create_masks(
                                                                input, output)
        
            predictions = self.transformer(input, 
                                    output,
                                    False,
                                    enc_padding_mask,
                                    dec_padding_mask,
                                    combined_mask
                                    )
            
            # select the last word from the seq_len dimension
            predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)

            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
            
            # return the result if the predicted_id is equal to the end token
            if predicted_id == end_token:
                result = tf.squeeze(output, axis=0)
            return comment_tokenizer.sequences_to_texts(output.numpy())[0]
            
            # concatentate the predicted_id to the output which is given to the decoder
            # as its input.
            output = tf.concat([output, predicted_id], axis=-1)

        result = tf.squeeze(output, axis=0)
        return comment_tokenizer.sequences_to_texts(output.numpy())[0]


if __name__ == '__main__':
    commentGenerator = CommentGenerator()
    print('yay')
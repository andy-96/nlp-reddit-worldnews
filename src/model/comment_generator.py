from tensorflow.keras import preprocessing
import tensorflow as tf
import os
import pickle

from src.config import PROCESSED_DATA_PATH, NUM_LAYERS, EMBEDDING_DIMS, NUM_HEADS, EXPANDED_DIMS
from src.model.transformer import Transformer
from src.utils import preprocess_sentence

class CommentGenerator():
    def __init__(self):

        self.transformer = Transformer(NUM_LAYERS,
                                       EMBEDDING_DIMS,
                                       NUM_HEADS,
                                       EXPANDED_DIMS,
                                       input_vocab_size,
                                       target_vocab_size,
                                       pe_input=input_vocab_size,
                                       pe_target=target_vocab_size)

    def max_len(self, sentence):
        return max(len(s) for s in sentence)

    def translate(self, sentence):
        sentence = preprocess_sentence(sentence)
        input = self.headline_tokenizer.texts_to_sequences([sentence])
        input = tf.convert_to_tensor(input)
        
        # as the target is German, the first word to the transformer should be the
        # German start token
        start_token = tf.convert_to_tensor([self.comment_tokenizer.word_index['<start>']])
        start_token = tf.expand_dims(start_token, 0)

        # And it should stop with the end token
        end_token = tf.convert_to_tensor(self.comment_tokenizer.word_index['<end>'])
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
                return self.comment_tokenizer.sequences_to_texts(output.numpy())[0]
            
            # concatentate the predicted_id to the output which is given to the decoder
            # as its input.
            output = tf.concat([output, predicted_id], axis=-1)

        return self.comment_tokenizer.sequences_to_texts(output.numpy())[0]


if __name__ == '__main__':
    commentGenerator = CommentGenerator()
    print('yay')
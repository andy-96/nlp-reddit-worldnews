import tensorflow as tf
import os
import argparse

from api.model.dataset import Dataset
from api.model.transformer import Transformer
from api.utils import preprocess_sentence, create_masks
from api.config import NUM_LAYERS, EMBEDDING_DIMS, NUM_HEADS, EXPANDED_DIMS, CKPT_PATH

class CommentGenerator():
    def __init__(self):
        self.dataset = Dataset()
        self.transformer = Transformer(NUM_LAYERS,
                                       EMBEDDING_DIMS,
                                       NUM_HEADS,
                                       EXPANDED_DIMS,
                                       self.dataset.input_vocab_size,
                                       self.dataset.target_vocab_size,
                                       pe_input=self.dataset.input_vocab_size,
                                       pe_target=self.dataset.target_vocab_size)
        weights_name = sorted([file for file in os.listdir(CKPT_PATH) if 'temp_model' in file], reverse=True)[0]
        weights_name = weights_name.split('.')[0]
        self.transformer.load_weights(os.path.join(CKPT_PATH, weights_name))

        self.headline_tokenizer = self.dataset.headline_tokenizer
        self.comment_tokenizer = self.dataset.comment_tokenizer
        self.max_length_output = self.dataset.max_length_output

    def generate(self, sentence):
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
            enc_padding_mask, dec_padding_mask, combined_mask, = create_masks(
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
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--headline', help='type your headline')
    args = parser.parse_args()

    commentGenerator = CommentGenerator()
    print(commentGenerator.generate(args.headline))
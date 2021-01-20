from tensorflow.keras import preprocessing
from api.ml.preprocessing import Preprocessing
import tensorflow as tf

def get_tokenized_headline():
    preprocessing = Preprocessing()
    headlines = preprocessing.headlines
    comments = preprocessing.comments

    headline_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    headline_tokenizer.fit_on_texts(headlines)

    comment_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    comment_tokenizer.fit_on_texts(comments)

    return headline_tokenizer, comment_tokenizer


def translate(sentence):
    preprocessing = Preprocessing()
    headline_tokenizer = get_tokenized_headline()

    sentence = preprocessing.preprocess_sentence(sentence)
    input = headline_tokenizer.texts_to_sequences([sentence])
    input = tf.convert_to_tensor(input)
    
    # as the target is German, the first word to the transformer should be the
    # German start token
    start_token = tf.convert_to_tensor([comment_tokenizer.word_index['<start>']])
    start_token = tf.expand_dims(start_token, 0)

    # And it should stop wiht the end token
    end_token = tf.convert_to_tensor(comment_tokenizer.word_index['<end>'])
    end_token = tf.expand_dims(end_token, 0)
    
    output = start_token
    for i in range(max_length_output):
        enc_padding_mask, dec_padding_mask, combined_mask, = create_masks(
                                                            input, output)
    
        predictions = transformer(input, 
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
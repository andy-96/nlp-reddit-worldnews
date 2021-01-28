import unicodedata
import re
import tensorflow as tf
import os
import yaml

from api.config import MODEL_PATH


def load_model_params(selected_model):
    pretrained_models = os.listdir(MODEL_PATH)
    for model in pretrained_models:
        if selected_model == model:
            config_path = os.path.join(MODEL_PATH, model, 'model_config.yaml')
            with open(config_path, 'r') as f:
                model_params = yaml.safe_load(f)
                f.close()
            break
    else:
        print('No model config found!')
        exit(1)

    num_layers = model_params['NUM_LAYERS']
    embedding_dims = model_params['EMBEDDING_DIMS']
    num_heads = model_params['NUM_HEADS']
    expanded_dims = model_params['EXPANDED_DIMS']
    epochs = model_params['EPOCHS']
    batch_size = model_params['BATCH_SIZE']
    
    return num_layers, embedding_dims, num_heads, expanded_dims, epochs, batch_size


def preprocess_sentence(w):
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


def create_masks(input, target):
    # Encoder padding mask
    encoder_padding_mask = create_padding_mask(input)
    
    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    decoder_padding_mask = create_padding_mask(input)
    
    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by 
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(target)[1])
    decoder_target_padding_mask = create_padding_mask(target)
    combined_mask = tf.maximum(decoder_target_padding_mask, look_ahead_mask)
    
    return encoder_padding_mask, decoder_padding_mask, combined_mask


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    
    # add extra dimensions to add the padding
    # to the attention logits. Tensor sizes are always a pain...
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask
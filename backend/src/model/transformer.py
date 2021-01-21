import tensorflow as tf
import numpy as np


class Transformer(tf.keras.Model):
    def __init__(self, num_layers, embedding_dims, num_heads, expanded_dims,
                 input_vocab_size, target_vocab_size, pe_input, pe_target,
                 rate=0.1):
        super().__init__()

        self.encoder = Encoder(num_layers, embedding_dims, num_heads,
                               expanded_dims, input_vocab_size, pe_input, rate)

        self.decoder = Decoder(num_layers, embedding_dims, num_heads,
                               expanded_dims, target_vocab_size, pe_target,
                               rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, input, target, training, encoder_padding_mask,
             decoder_padding_mask, look_ahead_mask):

        # (batch_size, inp_seq_len, embedding_dims)
        encoder_output = self.encoder(input, training, encoder_padding_mask)

        # (batch_size, target_seq_len, embedding_dims)
        dec_output = self.decoder(target, encoder_output, training,
                                  decoder_padding_mask, look_ahead_mask)

        # (batch_size, target_seq_len, target_vocab_size)
        final_output = self.final_layer(dec_output)

        return final_output


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, embedding_dims, num_heads, expanded_dims,
                 target_vocab_size, maximum_position_encoding, rate=0.1):
        super().__init__()

        self.embedding_dims = embedding_dims
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size,
                                                   embedding_dims)
        self.pos_encoding = positional_encoding(maximum_position_encoding,
                                                embedding_dims)

        self.decoder_layers = [DecoderLayer(embedding_dims, num_heads,
                                            expanded_dims, rate)
                               for i in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, encoder_output, training,
             padding_mask, look_ahead_mask):
        seq_len = tf.shape(x)[1]

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.embedding_dims, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.decoder_layers[i](x, encoder_output, training,
                                       padding_mask, look_ahead_mask)

        return x  # (batch_size, target_seq_len, d_model)


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, embedding_dims, num_heads, expanded_dims,
                 input_vocab_size, maximum_position_encoding, rate=0.1):
        super().__init__()

        self.embedding_dims = embedding_dims
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(
            input_vocab_size, embedding_dims)
        self.pos_encoding = positional_encoding(maximum_position_encoding,
                                                embedding_dims)

        self.dropout = tf.keras.layers.Dropout(rate)

        self.encoder_layers = [EncoderLayer(embedding_dims, num_heads,
                                            expanded_dims, rate)
                               for i in range(num_layers)]

    def call(self, x, training, padding_mask):
        seq_len = tf.shape(x)[1]

        x = self.embedding(x)
        # Technicality that  is used in the original paper
        x *= tf.math.sqrt(tf.cast(self.embedding_dims, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.encoder_layers[i](x, training, padding_mask)

        return x  # (batch_size, input_seq_len, embedding_dims)


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, dimensions):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(dimensions)[np.newaxis, :],
                            dimensions)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, embedding_dims, num_heads, expanded_dims, rate=0.1):
        super().__init__()

        self.mha1 = MultiHeadAttention(embedding_dims, num_heads)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.mha2 = MultiHeadAttention(embedding_dims, num_heads)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.ffn = pointwise_ffn(embedding_dims, expanded_dims)
        self.dropout3 = tf.keras.layers.Dropout(rate)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, encoder_output, training, padding_mask, look_ahead_mask):

        attn1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2 = self.mha2(encoder_output, encoder_output, out1, padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        # (batch_size, target_seq_len, embedding_dims)
        out3 = self.layernorm3(ffn_output + out2)

        return out3


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, embedding_dims, num_heads, expanding_dims, rate=0.1):
        super().__init__()

        self.mha = MultiHeadAttention(embedding_dims, num_heads)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.ffn = pointwise_ffn(embedding_dims, expanding_dims)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, training, padding_mask):

        attn_output = self.mha(x, x, x, padding_mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        # (batch_size, input_seq_len, embedding_dims)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, embedding_dims, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.embedding_dims = embedding_dims
        self.depth = embedding_dims // num_heads

        self.wq = tf.keras.layers.Dense(embedding_dims)
        self.wk = tf.keras.layers.Dense(embedding_dims)
        self.wv = tf.keras.layers.Dense(embedding_dims)

        self.dense = tf.keras.layers.Dense(embedding_dims)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, embedding_dims)
        k = self.wk(k)
        v = self.wv(v)

        # (batch_size, num_heads, seq_len, depth)
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # self_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        self_attention = calculate_self_attention(q, k, v, mask)

        # (batch_size, seq_len, num_heads, depth)
        self_attention = tf.transpose(self_attention, perm=[0, 2, 1, 3])

        # (batch_size, seq_len, embedding_dims)
        concat_attention = tf.reshape(
            self_attention, (batch_size, -1, self.embedding_dims))

        output = self.dense(concat_attention)

        return output


def pointwise_ffn(embedding_dims, expanded_dims):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(expanded_dims, activation='relu'),
        tf.keras.layers.Dense(embedding_dims)
    ])


def calculate_self_attention(q, k, v, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_scores = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_scores += (mask * -1e9)

    # softmax is normalized on the last axis (len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_scores, axis=-1)

    output = tf.matmul(attention_weights, v)

    return output

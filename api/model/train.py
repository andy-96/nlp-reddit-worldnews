import tensorflow as tf
import os
import argparse

from api.model.transformer import Transformer
from api.model.dataset import Dataset
from api.config import CKPT_PATH
from api.utils import create_masks, load_model_params

class Train():
    def __init__(self, selected_model):
        print('Initialize training')
        num_layers, embedding_dims, num_heads, \
            expanded_dims, self.epochs, _ = load_model_params(selected_model)
        
        self.dataset = Dataset(selected_model)
        self.transformer = Transformer(num_layers,
                                       embedding_dims,
                                       num_heads,
                                       expanded_dims,
                                       self.dataset.input_vocab_size,
                                       self.dataset.target_vocab_size,
                                       pe_input=self.dataset.input_vocab_size,
                                       pe_target=self.dataset.target_vocab_size)


        self.learning_rate = CustomSchedule(embedding_dims)
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate, beta_1=0.9, beta_2=0.98, 
                                     epsilon=1e-9)
        self.iterator = iter(self.dataset.dataset)

    def train_and_checkpoint(self):
        for fname in sorted(os.listdir(CKPT_PATH), reverse=True):
            if 'temp_model' in fname:
                filename = fname.split('.')[0]
                print(f'Use {filename}')
                self.transformer.load_weights(os.path.join(CKPT_PATH, filename))
                break

        for epoch in range(self.epochs):
            example = next(self.iterator)
            loss = self.train_step(epoch)
            if epoch % 1 == 0:
                self.transformer.save_weights(os.path.join(CKPT_PATH, f'{epoch}_temp_model'))
                print(f'Saved weights for step {epoch}')
                print("loss {:1.2f}".format(loss.numpy()))

    def train_step(self, epoch):
        epoch_loss = 0

        for (batch, (input, target)) in enumerate(self.dataset.dataset.take(self.dataset.steps_per_epoch)):
            decoder_input = target[ : , :-1 ] # ignore <end> token
            real = target[ : , 1: ]           # ignore <start> token
            
            enc_padding_mask, dec_padding_mask, combined_mask = create_masks(input, decoder_input)
            
            with tf.GradientTape() as tape:
                predictions = self.transformer(input, decoder_input, True, enc_padding_mask, dec_padding_mask, combined_mask)
                batch_loss = self.loss_function(real, predictions)

            gradients = tape.gradient(batch_loss, self.transformer.trainable_variables)    
            self.optimizer.apply_gradients(zip(gradients, self.transformer.trainable_variables))
            epoch_loss += batch_loss  

            if batch % 100 == 0:
                print ('Epoch {} Batch {} Loss {:.4f}'.format(
                    epoch + 1, batch, batch_loss.numpy()))

        return batch_loss
    
    def loss_function(self, real, pred):
        cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        loss = cross_entropy(y_true=real, y_pred=pred)
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        mask = tf.cast(mask, dtype=loss.dtype)
        loss = loss * mask
        loss = tf.reduce_mean(loss)
        return loss


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, embedding_dims, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        
        self.embedding_dims = embedding_dims
        self.embedding_dims = tf.cast(self.embedding_dims, tf.float32)

        self.warmup_steps = warmup_steps
        
    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        
        return tf.math.rsqrt(self.embedding_dims) * tf.math.minimum(arg1, arg2)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--create_new_model', help='Train new model')
    parser.add_argument('--model', help="Choose a model")
    args = parser.parse_args()

    train = Train(args.model)
    train.train_and_checkpoint()
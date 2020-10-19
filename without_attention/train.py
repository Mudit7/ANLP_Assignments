import tensorflow as tf
from keras_preprocessing import sequence
import numpy as np
from gensim.models import Word2Vec
import pandas as pd
import gensim.downloader as api
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from matplotlib import pyplot as plt
import os
from without_attention.Encoder import Encoder
from without_attention.Decoder import Decoder
# preprocessing
# corpus = api.load('text8')
# word2vec = Word2Vec(corpus)
# word2vec.save('gensim_w2v.emb')

START_INDEX = 0

training_data = np.load('training_data.npz', allow_pickle=True)
X = training_data['arr_0']
Y = training_data['arr_1']
english_vocab_size, hindi_vocab_size = training_data['arr_4']

max_sentence_len = 10
emb_dim = 50
latent_dim = 128
no_classes = hindi_vocab_size
BATCH_SIZE = 8

X = tf.keras.preprocessing.sequence.pad_sequences(X, padding='post', maxlen=max_sentence_len)
Y = tf.keras.preprocessing.sequence.pad_sequences(Y, padding='post', maxlen=max_sentence_len)

encoder = Encoder(emb_dim,english_vocab_size, latent_dim, BATCH_SIZE)
# example_input_batch = np.array([X[3]])
# sample_output, sample_hidden = encoder(example_input_batch)
decoder = Decoder(hindi_vocab_size, emb_dim, latent_dim, BATCH_SIZE)

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)



def train_step(inp, targ):
    loss = 0

    with tf.GradientTape() as tape:

        enc_output, enc_hidden = encoder(inp)

        # starts = tf.expand_dims([START_INDEX] * BATCH_SIZE, 1)
        # dec_input = tf.concat([starts,targ],axis = 1)
        predictions = decoder(targ, enc_hidden)
        loss += loss_function(targ, predictions)

    batch_loss = (loss / int(targ.shape[1]))

    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)

    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss


train_dataset = tf.data.Dataset.from_tensor_slices((X, Y))
print('number of samples = ',train_dataset.__len__().numpy())
train_dataset = train_dataset.batch(BATCH_SIZE)

EPOCHS = 10
losses = []
for epoch in range(EPOCHS):

    total_loss = 0

    for batch,(inp, targ) in enumerate(train_dataset):
        x = np.array(inp)
        y = np.array(targ)
        batch_loss = train_step(x, y)
        # print('batch_loss', batch_loss)
        total_loss += batch_loss
        losses.append(batch_loss)
    print(total_loss)
checkpoint_dir = './training_checkpoints/'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)
checkpoint.save(file_prefix = checkpoint_prefix)
plt.plot(losses)
plt.show()
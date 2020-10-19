import tensorflow as tf


class Encoder(tf.keras.Model):
    def __init__(self, embedding_dim, vocab_size, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.enc_units = enc_units
        self.batch_sz = batch_sz
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True)
        self.lstm = tf.keras.layers.LSTM(self.enc_units,
                                                     dropout=0.3,
                                                     return_state=True,
                                                     recurrent_activation='tanh',
                                                     recurrent_initializer='glorot_uniform')

    def call(self, x):
        x = self.embedding(x)
        final_output, state_h, state_c = self.lstm(x)
        return final_output, [state_h,state_c]


if __name__ == '__main__':
    vocab_size = 2000
    embedding = 200
    units = 120

    encoder = Encoder(embedding, vocab_size, units)
    dummy_in = tf.zeros((2, 15))
    # dummy_in[2]=1
    out1, out2 = encoder(dummy_in)
    print(out1, out2)

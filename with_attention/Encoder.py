import tensorflow as tf


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.enc_units = enc_units
        self.batch_sz = batch_sz
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True)
        self.bilstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM
                                                    (self.enc_units,
                                                     dropout=0.3,
                                                     return_sequences=True,
                                                     return_state=True,
                                                     recurrent_activation='tanh',
                                                     recurrent_initializer='glorot_uniform'), merge_mode='concat',
                                                    name="bi_lstm_0")

    def call(self, x):

        x = self.embedding(x)
        final_output, forward_h, forward_c, backward_h, backward_c = self.bilstm(x)
        # state_h = tf.math.add(forward_h, backward_h)
        return final_output, backward_h


if __name__ == '__main__':
    vocab_size = 2000
    embedding = 200
    units = 120

    encoder = Encoder(embedding, vocab_size, units)
    dummy_in = tf.zeros((2, 15))
    # dummy_in[2]=1
    out1, out2 = encoder(dummy_in)
    print(out1, out2)

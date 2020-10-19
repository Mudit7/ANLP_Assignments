import tensorflow as tf
from with_attention.Attention import BahdanauAttention


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.lstm = tf.keras.layers.LSTM(self.dec_units,return_sequences=True, recurrent_initializer='glorot_uniform')
        # self.gru = tf.keras.layers.GRU(self.dec_units,return_sequences=True, recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)
        # used for attention
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, dec_in, dec_hidden):
        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(dec_in)

        output = self.lstm(x,initial_state=dec_hidden)
        # output shape == (batch_size * 1, hidden_size)
        # output = tf.reshape(output, (-1, output.shape[2]))
        # output shape == (batch_size, vocab)
        x = self.fc(output)

        return x

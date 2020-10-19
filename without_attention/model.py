import tensorflow as tf
from bilsm_attention.Encoder import  Encoder
from bilsm_attention.Attention import BahdanauAttention
from bilsm_attention.Decoder import Decoder
import numpy as np

class BiLSTM_Attention(tf.keras.Model):
    def __init__(self, embedding_dim, units, nClasses,maxSentenceLength):
        super(BiLSTM_Attention, self).__init__()
        self.encoder = Encoder(embedding_dim, units)
        self.decoder = Decoder(nClasses)
        self.attention = BahdanauAttention(units)
        self.maxSentenceLength = maxSentenceLength

    def call(self, x):
        x = tf.convert_to_tensor(x)
        final_output, state_h = self.encoder(x)
        dec_hidden_state = state_h
        dec_input = [0] #start symbol
        for i in range(self.maxSentenceLength):
            context_vector, attention_weights = self.attention(final_output, dec_hidden_state)
            dec_input,dec_hidden_state = self.decoder(dec_input,context_vector)
        return x

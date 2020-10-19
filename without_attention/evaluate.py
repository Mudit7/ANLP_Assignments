import pickle
import tensorflow as tf
from keras_preprocessing import sequence
from without_attention.Encoder import Encoder
from without_attention.Decoder import Decoder
import numpy as np

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

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
english_idx2word = load_obj('english_idx2word')
hindi_idx2word = load_obj('hindi_idx2word')
encoder = Encoder(emb_dim,english_vocab_size, latent_dim, BATCH_SIZE)
# example_input_batch = np.array([X[3]])
# sample_output, sample_hidden = encoder(example_input_batch)
decoder = Decoder(hindi_vocab_size, emb_dim, latent_dim, BATCH_SIZE)
def evaluate(inputs):

  # inputs = tf.convert_to_tensor(inputs)

  result = []

  enc_out, enc_hidden = encoder(inputs)
  # starts = tf.expand_dims([START_INDEX] * BATCH_SIZE, 1)
  # dec_input = tf.concat([starts,targ],axis = 1)
  predictions = decoder(targ, enc_hidden)
  predictions = tf.reshape(tf.argmax(predictions,axis=-1),(BATCH_SIZE,max_sentence_len)).numpy()

  result = hindi_idx2word[predictions[0]]
  for i in range(1,len(max_sentence_len)):
    if predictions[i]==predictions[i-1]:
      break
    result.append(hindi_idx2word[predictions[i]])

  return result


inp = np.array(tf.slice(X,[0,0],[8,max_sentence_len]))
print(inp)
out = evaluate(inp)
inp_sentence = [hindi_idx2word[i] for i in Y[num] if i!=0]
print(inp_sentence,out)


for num in range(20):
  inp = np.array([X[num]])
  out = evaluate(inp)
  inp_sentence = [hindi_idx2word[i] for i in Y[num] if i!=0]
  # print(inp_sentence,out)
  BLEUscore = nltk.translate.bleu_score.sentence_bleu([inp_sentence], out, weights = (0.5, 0.5))
  print(BLEUscore)
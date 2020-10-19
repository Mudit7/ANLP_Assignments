import tensorflow as tf
import numpy as np
import gensim
from gensim import corpora
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.text import text_to_word_sequence
import pandas as pd
from tokenize import tokenize
from gensim.utils import tokenize
from inltk.inltk import Tokenizer as hindi_tokenizer

max_padded_seq_size = 30
english_emb_size = 100
hindi_emb_size = 100

def dataloader(max_len):
    def preprocess(w2v,text,language):
        embeddings = []
        if language == 'english':
            text = text.replace("'",'')
            words = text_to_word_sequence(text)
        elif language == 'hindi':
            text = text.replace(",", '')
            text = text.replace("|", ' ')
            words = text.split()
        else:
            raise Exception("Choose lang as 'hindi' or 'english' ")
        for word in words:
            if word in w2v:
                embeddings.append(w2v[word])

        cur_seq_len = len(embeddings)
        print(words)
        print(cur_seq_len,language)
        # print(text)
        if cur_seq_len < max_len:
            embeddings = np.pad(embeddings, [(0, max_len - cur_seq_len), (0, 0)])
        else:
            embeddings = embeddings[cur_seq_len - max_len:]

        return embeddings

    hindi_wv = Word2Vec.load('hindi_100.emb')
    english_wv = Word2Vec.load('english_100.emb')
    df = pd.read_pickle('en_hi.pkl')
    df.reset_index(drop=True,inplace=True)
    x_train = []
    y_train = []
    for ind in df.index:
        print(ind)
        x_train.append(preprocess(english_wv,df['english'][ind],language='english'))
        y_train.append(preprocess(hindi_wv,df['hindi'][ind],language='hindi'))

    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    np.savez('processed_data.npz', x_train, y_train)
    print('saved.')
    return

# dataloader(max_padded_seq_size)
df = pd.read_pickle('en_hi.pkl')
df.reset_index(drop=True,inplace=True)

print(df[100])
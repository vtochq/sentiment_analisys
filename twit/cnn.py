import pandas as pd
#import codecs
import my_lib
import datetime
import pickle
import numpy as np

from sklearn.model_selection import train_test_split
from time import time

from bs4 import BeautifulSoup
import re

from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from gensim.models.keyedvectors import KeyedVectors

from tensorflow.keras.layers import Dense, Input, GlobalMaxPooling1D
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Conv2D, MaxPooling2D, Dropout, concatenate, Activation
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Flatten
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from nltk.tokenize import WordPunctTokenizer

from tensorflow.keras.callbacks import ModelCheckpoint

tok = WordPunctTokenizer()
pat1 = r'@[A-Za-z0-9]+'
pat2 = r'https?://[A-Za-z0-9./]+'
combined_pat = r'|'.join((pat1, pat2))


def debug(text):
    print(datetime.datetime.now().replace(microsecond=0), " ", text)

def tweet_cleaner(text):
    soup = BeautifulSoup(text, 'html.parser')
    souped = soup.get_text()
    stripped = re.sub(combined_pat, '', souped)
    try:
        clean = stripped.decode("utf-8-sig").replace(u"\ufffd", "?")
    except:
        clean = stripped
    letters_only = re.sub("[^a-zA-Zа-яА-Я_]", " ", clean)

    words = tok.tokenize(letters_only)
    return (" ".join(words)).strip()

infprefix = "100_"
#infprefix = ""
SEED = 20
EMBEDDING_DIM=500
NUM_WORDS=200000

debug("Loading Positive")
dataset = pd.DataFrame()

data_raw = pd.read_csv(infprefix+'positive.csv', header=None, encoding='utf8', sep=';')[3]
dataset['Words'] = data_raw
dataset['Cat'] = 1


debug("Loading Negative")
temp = pd.DataFrame()

data_raw = pd.read_csv(infprefix+'negative.csv', header=None, encoding='utf8', sep=';')[3]
temp['Words'] = data_raw
temp['Cat'] = 0

dataset = dataset.append(temp, ignore_index=True)

temp=None

dataset_x = dataset['Words'].str.lower().values
dataset_y = dataset['Cat'].values

dataset_x_cleaned = []
for i in range(0,len(dataset_x)):
    dataset_x_cleaned.append(tweet_cleaner(dataset_x[i]))


x_train, x_validation_and_test, y_train, y_validation_and_test = train_test_split(dataset_x_cleaned, dataset_y, test_size=.1, random_state=SEED)

x_validation, x_test, y_validation, y_test = train_test_split(x_validation_and_test, y_validation_and_test, test_size=.5, random_state=SEED)


debug("Start tokenazing")
tokenizer = Tokenizer(num_words=NUM_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\'',lower=True, char_level=False)
tokenizer.fit_on_texts(x_train)
debug("Tokenazing learn data")
tx_train = tokenizer.texts_to_sequences(x_train)
debug("Tokenazing validation data")
tx_validation = tokenizer.texts_to_sequences(x_validation)
debug("Tokenazing test data")
tx_test = tokenizer.texts_to_sequences(x_test)

print('Found %s unique tokens.' % (len(tokenizer.word_index)+1))
word_dict = tokenizer.word_index

debug("Padding learn data")
tx_train = pad_sequences(tx_train, value=0, padding='post')  # , maxlen=256

max_word_count = len(max(tx_train, key=len))

debug("Padding validation data")
tx_validation = pad_sequences(tx_validation, value=0, padding='post', maxlen=max_word_count)

debug("Padding test data")
tx_test = pad_sequences(tx_test, value=0, padding='post', maxlen=max_word_count)



debug("Loading word2vec")
word_vectors = KeyedVectors.load_word2vec_format('../../all.norm-sz500-w10-cb0-it3-min5.w2v', binary=True, unicode_errors='ignore')
vocabulary_size = min(len(word_dict)+1,NUM_WORDS)

debug("Zeroing embedding matrix")
embedding_matrix = np.zeros((vocabulary_size, EMBEDDING_DIM))
not_found_in_vec=0
debug("Filling embedding matrix")
for word, i in word_dict.items():
    if i>=vocabulary_size:
        continue
    try:
        embedding_matrix[i] = word_vectors[word]
    except KeyError:
        embedding_matrix[i]=np.random.normal(0,0.5,EMBEDDING_DIM)
        not_found_in_vec=not_found_in_vec+1

print("Words not in w2v: ", not_found_in_vec)

tweet_input = Input(shape=(max_word_count,), dtype='int32')

tweet_encoder = Embedding(vocabulary_size, EMBEDDING_DIM,  input_length=max_word_count, weights=[embedding_matrix], trainable=True)(tweet_input) # weights=[embedding_matrix],
bigram_branch = Conv1D(filters=100, kernel_size=2, padding='valid', activation='relu', strides=1)(tweet_encoder)
bigram_branch = GlobalMaxPooling1D()(bigram_branch)
trigram_branch = Conv1D(filters=100, kernel_size=3, padding='valid', activation='relu', strides=1)(tweet_encoder)
trigram_branch = GlobalMaxPooling1D()(trigram_branch)
fourgram_branch = Conv1D(filters=100, kernel_size=4, padding='valid', activation='relu', strides=1)(tweet_encoder)
fourgram_branch = GlobalMaxPooling1D()(fourgram_branch)
merged = concatenate([bigram_branch, trigram_branch, fourgram_branch], axis=1)

merged = Dense(256, activation='relu')(merged)
merged = Dropout(0.5)(merged)
merged = Dense(1)(merged)
output = Activation('sigmoid')(merged)
model = Model(inputs=[tweet_input], outputs=[output])
model.compile(loss='binary_crossentropy',
                  optimizer='Nadam',
                  metrics=['accuracy'])
model.summary()

filepath="CNN_best_weights_04_10_18_16_44.{epoch:02d}-{val_acc:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
model.fit(tx_train, y_train, batch_size=512, epochs=5, validation_data=(tx_validation, y_validation), callbacks = [checkpoint])

model.evaluate(tx_test, y_test)

model.predict(pad_sequences(tokenizer.texts_to_sequences(["образец текста"]), value=0, padding='post', maxlen=max_word_count))

from sklearn.metrics import classification_report, confusion_matrix

# Confusion matrix
pred = model.predict(tx_test)
binpred = [0 if n <= 0.5 else 1 for n in pred]
conmat = np.array(confusion_matrix(y_test, binpred, labels=[1,0]))
confusion = pd.DataFrame(conmat, index=['positive', 'negative'], columns=['predicted_positive','predicted_negative'])
print(confusion)

###

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

cvec = CountVectorizer()
lr = LogisticRegression()
cvec.set_params(max_features=200000, ngram_range=(1, 3))
checker_pipeline = Pipeline([
            ('vectorizer', cvec),
            ('classifier', lr)
        ])

sentiment_fit = checker_pipeline.fit(x_train, y_train)
y_pred = sentiment_fit.predict(x_test)
accuracy_score(y_test, y_pred)

import numpy as np
import pandas as pd
import string

import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Embedding, Dense, LSTM, Flatten, Bidirectional, Dropout
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer

from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("final_data.csv",  encoding='utf-8')


def preprocess(text):
    stopword = stopwords.words('english')
    text = text.lower()
    text = "".join([i for i in text if i not in string.punctuation])
    text = text.split()
    text = [i for i in text if i not in stopword]
    wordnet_lem = WordNetLemmatizer()
    text = [wordnet_lem.lemmatize(word) for word in text]
    return text


X = list(map(preprocess, df['Text']))
y = df['Label']


label_enc = LabelEncoder()
label_enc.fit(y)
y = label_enc.transform(y)


max_len = 32

token = Tokenizer()
token.fit_on_texts(X)

word_index = token.word_index
vocab_size = len(word_index) + 1000
print(vocab_size)

X_changed = token.texts_to_sequences(X)
X_sample = pad_sequences(X_changed, maxlen=max_len, padding='post', truncating='post')

X_sample = np.asarray(X_sample)
y = np.asarray(y)
y = np.reshape(y, (-1, 1))

model = Sequential([
    Embedding(vocab_size, 16, input_length=max_len),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.2),
    Bidirectional(LSTM(64)),
    Dropout(0.2),
    Flatten(),
    Dense(24, activation='relu'),
    Dense(1, activation='sigmoid')
])


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
print(model.summary())

history = model.fit(X_sample, y, epochs=10, batch_size=32)
model.save('model')













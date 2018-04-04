import pandas as pd
import numpy as np
from glob import glob
import os
import sys
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.utils import resample

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split, GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers.core import Activation
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, RNN
from keras.layers.embeddings import Embedding

TRAIN_PATH = 'data/train'
TEST_PATH = 'data/test'
SEED = 2018

def get_x_y(file_path):
    # Add a "cache" check here
    files = {}
    files['pos'] = glob(os.path.join(file_path, 'pos', '*.txt'))
    files['neg'] = glob(os.path.join(file_path, 'neg', '*.txt'))
    sentiment_map = {'pos': 1, 'neg': 0}
    x = []
    y = []
    for sentiment in files:
        for file_name in files[sentiment]:
            temp_ = []
            with open(file_name) as file_:
                temp_ = file_.read()
            x.append(temp_)
            y.append(sentiment_map[sentiment])
    return x, y


def prep_data(vocab_size=100, max_review_len=150, n_samples=5000):
    x_train_raw, y_train_raw = get_x_y(TRAIN_PATH)
    x_test_raw, y_test_raw = get_x_y(TEST_PATH)
    
    time_start = datetime.now()
    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(x_train_raw)
    
    # Fit our training data
    x_train_sequence = tokenizer.texts_to_sequences(x_train_raw)
    x_train_pad = pad_sequences(x_train_sequence, maxlen=max_review_len)

    # Fit our testing data
    x_test_sequence = tokenizer.texts_to_sequences(x_test_raw)
    x_test_pad = pad_sequences(x_test_sequence, maxlen=max_review_len)

    # Subset for testing
    x_train_pad_sub, y_train_sub = resample(x_train_pad, y_train_raw, replace=False, n_samples=n_samples, random_state=SEED)
    x_test_pad_sub, y_test_sub = resample(x_test_pad, y_test_raw, replace=False, n_samples=n_samples, random_state=SEED)
    
    return x_train_pad_sub, y_train_sub, x_test_pad_sub, y_test_sub


def basic_lstm_model(
    embedding_vector_length=32,
    dropout_rate=0.2, 
    vocab_size=100, 
    max_review_len=150,
    lstm_len=100
):
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_vector_length, input_length=max_review_len))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(lstm_len))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def train_and_evaluate_model(
    x_train,
    y_train,
    x_test,
    y_test,
    num_epochs=5,
    batch_size=32,
    max_review_len=100,
    embed_length=32,
    lstm_len=100,
    vocab_size=100,
    dropout_rate=0.2,
    verbose=1
):
    time_start = datetime.now()
    model = basic_lstm_model(
        embedding_vector_length=embed_length,
        dropout_rate=dropout_rate,
        vocab_size=vocab_size,
        max_review_len=max_review_len,
        lstm_len=lstm_len
    )
    model.fit(x_train, y_train, epochs=num_epochs, batch_size=batch_size, verbose=verbose)
    scores = model.evaluate(x_test, y_test)
    return scores[1]*100

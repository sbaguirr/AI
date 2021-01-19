import joblib
import os

import gensim
import keras
from keras.preprocessing.sequence import pad_sequences
import pandas as pd

from .preprocesser import preprocess
from .representer import represent_tweets


class LogisticRegressionClassifier():
    def __init__(self):
        module_dir = os.path.dirname(__file__)
        self.models_dir = os.path.join(module_dir, 'models')
        self.load_wikipedia_embeddings()
        self.load_model()
        super().__init__()

    def load_model(self):
        model_path = os.path.join(self.models_dir, 'logit', 'logit_3classes_word2vec_17_01.model')
        self.model = joblib.load(model_path)

    def load_wikipedia_embeddings(self):
        path = os.path.join(self.models_dir, 'word_embeddings', 'wikipedia_es_300_sg.bin')
        self.wikipedia = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)

    def load_data(self, tweets):
        data = []
        for tweet in tweets:
            tweet_id = tweet.get('id_url')
            tweet = tweet.get('full_text')
            data.append((tweet_id, tweet))
        return pd.DataFrame(data=data, columns=['id', 'tweet'])

    def classify(self, tweets):
        data = self.load_data(tweets)
        data.tweet = data.tweet.apply(preprocess)
        data = data[~data.tweet.isna()]
        features = represent_tweets(data, self.wikipedia)
        data['label'] = self.model.predict(features)
        return { label: list(tweets.id) for label, tweets in data.groupby('label') }


class LSTMClassifier():
    def __init__(self):
        module_dir = os.path.dirname(__file__)
        self.models_dir = os.path.join(module_dir, 'models')
        self.load_model()
        super().__init__()

    def load_model(self):
        model_path = os.path.join(self.models_dir, 'lstm', 'model.h5')
        self.model = keras.models.load_model(model_path)
        tokenizer_path =os.path.join(self.models_dir, 'lstm', 'model.tokenizer')
        self.tokenizer = joblib.load(tokenizer_path)

    def load_data(self, tweets):
        data = []
        for tweet in tweets:
            tweet_id = tweet.get('id_url')
            tweet = tweet.get('full_text')
            data.append((tweet_id, tweet))
        return pd.DataFrame(data=data, columns=['id', 'tweet'])

    def classify(self, tweets):
        data = self.load_data(tweets)
        data.tweet = data.tweet.apply(preprocess)
        data = data[~data.tweet.isna()]
        predict_sequences = self.tokenizer.texts_to_sequences(data.tweet)
        predict_padded = pad_sequences(predict_sequences, maxlen=20, padding='post', truncating='post')
        data['label'] = self.model.predict_classes(predict_padded, batch_size=1)
        return { label: list(tweets.id) for label, tweets in data.groupby('label') }

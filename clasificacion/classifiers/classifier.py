import joblib
import os

import gensim
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
        model_path = os.path.join(self.models_dir, 'logit_14-01-2020.model')
        self.model = joblib.load(model_path)

    def load_wikipedia_embeddings(self):
        path = os.path.join(self.models_dir, 'wikipedia_es_300_sg.bin')
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

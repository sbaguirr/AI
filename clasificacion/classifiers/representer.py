import numpy as np
import pandas as pd


def doc2vec(doc, embeddings):
    doc = doc.split()
    new_doc = [word for word in doc if word in embeddings.wv]
    if len(new_doc) > 0:
        return embeddings.wv[new_doc].mean(axis=0)
    return np.array([np.nan]*300)

def represent_tweets(data, embeddings):
    vecs = data.tweet.apply(lambda doc: doc2vec(doc, embeddings))
    vecs = pd.DataFrame(index=data.index, data=vecs.values.tolist())
    return vecs.dropna()

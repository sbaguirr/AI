import numpy as np
import pandas as pd


def doc2vec(doc, embeddings):
    doc = doc.split()
    new_doc = [word for word in doc if word in embeddings.wv]
    if len(new_doc) > 0:
        return embeddings.wv[new_doc].mean(axis=0)
    return np.array([np.nan]*300)

def docs2vec(data, embeddings):
    vecs = data.tweet.apply(lambda doc: doc2vec(doc, embeddings))
    vecs = pd.DataFrame(index=data.index, data=vecs.values.tolist())
    return vecs.dropna()

def docs_to_tfidf(data, pca_model, tfidf_model):
    vectors = tfidf_model.transform(data.tweet.values).todense()
    vectors = pca_model.transform(vectors)
    return pd.DataFrame(index=data.index, data=vectors)

def represent_tweets(data, embeddings, pca_model, tfidf_model):
    embs = docs2vec(data, embeddings)
    tfidf_reduced = docs_to_tfidf(data, pca_model, tfidf_model)
    return embs.merge(tfidf_reduced, left_index=True, right_index=True)

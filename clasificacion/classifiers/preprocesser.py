import re
import string

import nltk
import numpy as np


regex_1 = re.compile(r'\S+(\.)(com|net|ly|co|us|ec|gob)(\S?)+')
regex_2 = re.compile(r'(http|facebook|twitter|bit|soundcloud|www|pic|#|@)\S+')

def normalize_case(data):
    return data.lower()

def remove_hastags_mentions_links(data):
    data = regex_1.sub('', data)
    data = regex_2.sub('', data)
    return data

def remove_punctuations(data):
    trans_table = str.maketrans('', '', string.punctuation)
    return [token.translate(trans_table) for token in data]

def remove_numerics(data):
    return [token for token in data if token.isalpha()]

def remove_short_lines(data, short_line_words=0):
    return data if len(data) > short_line_words else np.nan

def tokenize(data):
    return nltk.word_tokenize(data, language='spanish')

def undo_tokenization(data):
    if data is np.nan:
        return np.nan
    return ' '.join(data)

def preprocess(data):
    data = normalize_case(data)
    data = remove_hastags_mentions_links(data)
    data = tokenize(data)
    data = remove_punctuations(data)
    data = remove_numerics(data)
    data = remove_short_lines(data, 3)
    data = undo_tokenization(data)
    return data

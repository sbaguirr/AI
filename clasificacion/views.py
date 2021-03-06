from django.shortcuts import render

import tweepy

from .classifiers.classifier import LogisticRegressionClassifier
from .classifiers.classifier import LSTMClassifier


consumer_key = ''
consumer_secret = ''
access_token = ''
access_token_secret = ''

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)

logit_classifier = LogisticRegressionClassifier()
lstm_classifier = LSTMClassifier()

def index(request):
    return render(request, 'clasificacion/index.html')

def hashtag(request):
    if request.method == 'POST':
        hasht = request.POST['hashtag']
        query = f'{hasht} -filter:retweets'
        date = f"{request.POST['anioDesde']}-{request.POST['mesDesde']}-{request.POST['diaDesde']}"

        data = []
        cursor = tweepy.Cursor(api.search, tweet_mode='extended', q=query, count=100, lang='es', since=date)
        for tweet in cursor.items(int(request.POST['cantidad'])):
            id_str = tweet._json['id_str']
            tweet._json['id_url'] = f'https://twitter.com/twitter/statuses/{id_str}'
            data.append(tweet._json)

        if request.POST['model'] == 'lg' :
            prediction = logit_classifier.classify(data)
        else:
            prediction = lstm_classifier.classify(data)

        if len(prediction) == 0:
            context = { 'info': 'No existen tweets que cumplan con los parámetros de búsqueda ingresados.' }
        else:
            context = {
                'resultados': True,
                'pedidosAyuda': prediction.get(0, []),
                'ofertas': prediction.get(1, []),
                'ninguna': prediction.get(2, [])
            }
        return render(request, 'clasificacion/hashtag.html', context)
    else:
        return render(request, 'clasificacion/hashtag.html')

def tweet(request):
    return render(request, 'clasificacion/tweet.html')

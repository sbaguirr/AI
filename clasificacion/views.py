from django.shortcuts import render
import tweepy
consumer_key = ''
consumer_secret = ''
access_token = ''
access_token_secret = ''
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)


# Create your views here.
def index(request):
    return render(request, "clasificacion/index.html")


def hashtag(request):
    if request.method == "POST":
        hasht = request.POST["hashtag"]
        fecha = request.POST["anioDesde"]+'-'+ request.POST["mesDesde"]+'-'+request.POST["diaDesde"]
        pedidosAyuda = []
        quejas = []
        ofertas = []
        ninguna= []
        cursor = tweepy.Cursor(api.search, tweet_mode="extended", q=hasht, count=100, lang="es", since= fecha)
        for tweet in cursor.items(int(request.POST["cantidad"])):
            ninguna.append('https://twitter.com/twitter/statuses/'+str(tweet.id)+'?ref_src=twsrc%5Etfw')
        return render(request, "clasificacion/hashtag.html", {
            "pedidosAyuda": pedidosAyuda,
            "quejas": quejas,
            "ofertas": ofertas,
            "ninguna": ninguna
        })
    else:
        return render(request, "clasificacion/hashtag.html")


def tweet(request):
    return render(request, "clasificacion/tweet.html")

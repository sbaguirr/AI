from django.shortcuts import render


# Create your views here.
def index(request):
    return render(request, "clasificacion/index.html")

def hashtag(request):
    return render(request, "clasificacion/hashtag.html")    

def tweet(request):
    return render(request, "clasificacion/tweet.html")    
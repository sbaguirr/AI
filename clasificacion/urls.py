from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
     path("hashtag", views.hashtag, name="hashtag"),
      path("tweet", views.tweet, name="tweet"),
]

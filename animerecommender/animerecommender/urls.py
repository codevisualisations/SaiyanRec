"""animerecommender URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from . import views #go to the current directory, and grab all the views from there
from django.conf import settings


urlpatterns = [
    path('', views.HomePage.as_view(), name="home"),#registering the templates and views so when a user clicks its clear which webpage to be directed to
    path('admin/', admin.site.urls),
    path('test/', views.TestPage.as_view(), name="test"),
    path('thanks/', views.ThanksPage.as_view(), name="thanks"),
    path('accounts/', include("accounts.urls", namespace="accounts")), #connects the accounts namespace to acccounts.url. So if someone logs in or signs up
    # it connects to the urls.py file, and the accounts application I created
    path('accounts/', include("django.contrib.auth.urls")),#allows us to connect everything django has under the hood for authorisation. Connects to the login pages for the accounts
    path('posts/', include("posts.urls", namespace="posts")),
    path('groups/',include("groups.urls", namespace="groups")),
    path('recommend/', views.recommend, name='recommend'),
    path('result/', views.result, name='result'),
    path('recommend2/', views.recommend2, name='recommend2'),
    path('result2/', views.result2, name='result2'),
    path('result3/', views.my_view, name='result3'),
    path('result4/', views.newview, name='result4'),
    path('result5/', views.newview2, name='result5')
    # path('recommend3/', views.recommend3, name='recommend3'),
    # path('result3/', views.result3, name='result3'),
]

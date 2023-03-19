from django.urls import path
from django.contrib.auth import views as auth_views #with the integration of the log-in and log-out view,
#we no longer have to take care of this manually in the views.py
from . import views #import the views.py file (mine own views, this is not to be confused with auth_views)

app_name = 'accounts'

urlpatterns = [
    path('login/', auth_views.LoginView.as_view(template_name="accounts/login.html"),name='login'),
    path('logout/', auth_views.LogoutView.as_view(), name="logout"),
    path('signup/', views.SignUp.as_view(), name="signup"),
]

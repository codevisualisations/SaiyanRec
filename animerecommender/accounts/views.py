from django.contrib.auth import login, logout
from django.urls import reverse_lazy #this is used in case someone is logged in or logged out, it shows where they should go
from django.views.generic import CreateView #for creating a new user
from . import forms #forms are imported here since the forms for logging in and signing up need to be connected to these views

# Create your views here.
class SignUp(CreateView):
    form_class = forms.UserCreateForm
    success_url = reverse_lazy("login") #once someone has signed up, on a successful signup, we will take them back to the login page.
    #it's called lazy since we do not want this to execute UNTIL they've hit submit on the signup button
    template_name = "accounts/signup.html"

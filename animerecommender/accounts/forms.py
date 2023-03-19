from django.contrib.auth import get_user_model #returns the currently active user model in the project
from django.contrib.auth.forms import UserCreationForm #there's already a user creation form built in to django


class UserCreateForm(UserCreationForm):#in the brackets, we're inherting from the django import
    class Meta:
        fields = ("username", "email", "password1", "password2")#these fields are already available from contrib.auth
        model = get_user_model() #allows us to retrieve the model of whoever is accessing the website

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["username"].label = "User ID"
        self.fields["email"].label = "Email address"

#So, when a user comes in and is ready to sign up,
#user creation form is called, from auth.forms. The meta class dictates the fields
#we need filled in from a user

#def init initialises, and adds labels to these forms. It's similar to doing it from a HTML page, but here we're just doing it from forms.py

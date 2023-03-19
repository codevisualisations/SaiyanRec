from django.db import models
from django.contrib import auth #alot of the authorisation tools for accounts are built in to Django, this way we don't have to mess around with creating our own models for users etc.
from django.utils import timezone

# Creates a simple model for accounts using Django's built in 'models'
class User(auth.models.User, auth.models.PermissionsMixin):

    def __str__(self): #string representation of the User object
        return "@{}".format(self.username)
        #this is essentially a string representation of my newly created object
        #if we want a string representation of a user, do the following
        #username is a built in function. it comes included in auth.models.user, hence why we did not need to define it

#making migrations creates the given model. In this case, it creates user

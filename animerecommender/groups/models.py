#SOURCE: sections of code in this file are all taken from one source => https://www.udemy.com/course/python-and-django-full-stack-web-developer-bootcamp/learn/lecture/6649278?start=615#content
from django.db import models
from django.conf import settings
from django.urls import reverse
from django.utils.text import slugify #allows us to remove characters that aren't alphanumeric e.g !,-,£ etc
# from accounts.models import User

# pip install misaka
import misaka #allows link embedding

from django.contrib.auth import get_user_model #returns the currently active user model in the project
User = get_user_model() #call things off of the current users session

# https://docs.djangoproject.com/en/2.0/howto/custom-template-tags/#inclusion-tags
# This is for the in_group_members check template tag
from django import template #this is how we can use custom template tags
register = template.Library()


class Group(models.Model):#attributes for our group
    name = models.CharField(max_length=255, unique=True)#name of the group
    slug = models.SlugField(allow_unicode=True, unique=True)#slug representation of the group
    description = models.TextField(blank=True, default='')#it's group description
    description_html = models.TextField(editable=False, default='', blank=True)
    members = models.ManyToManyField(User,through="GroupMember")#calls all the members belonging to a particular group
#methods for the group, what it can do
    def __str__(self):
        return self.name #show the name of the group

    def save(self, *args, **kwargs):
        self.slug = slugify(self.name)#the name of the group entered by the user is slugified. i.e lowercased and spaces removed
        self.description_html = misaka.html(self.description)
        super().save(*args, **kwargs)

    def get_absolute_url(self):
        return reverse("groups:single", kwargs={"slug": self.slug})


    class Meta:
        ordering = ["name"]


class GroupMember(models.Model):#connects to group
    group = models.ForeignKey(Group,related_name='memberships',on_delete=models.CASCADE)
    #the group member is related to the group class, through the foreign key called memberships
    user = models.ForeignKey(User,related_name='user_groups',on_delete=models.CASCADE)
    #a link to user, and related name is user_groups. The current logged in user, is a member of
    #certain groups. Group member class is linked to both the User, and various groups they could belong to
    def __str__(self):
        return self.user.username
        #string representation of this object
    class Meta:
        unique_together = ("group", "user")

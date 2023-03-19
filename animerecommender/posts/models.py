from django.db import models
from django.conf import settings
from django.urls import reverse #when someone does a post, we can redirec them to a certain screen
#SOURCE: sections of code in this file are all taken from one source => https://www.udemy.com/course/python-and-django-full-stack-web-developer-bootcamp/learn/lecture/6649278?start=615#content

import misaka

from groups.models import Group #so we can connect a post, to a group

from django.contrib.auth import get_user_model
User = get_user_model() #connects the current post, to whoever is logged in as a user


class Post(models.Model):
    user = models.ForeignKey(User, related_name="posts",on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now=True)#once someone posts, the date and time of this action is recorded
    message = models.TextField()
    message_html = models.TextField(editable=False)
    group = models.ForeignKey(Group, related_name="posts",null=True, blank=True,on_delete=models.CASCADE)
    #^ post is connected with a foreign key to a group
    def __str__(self):
        return self.message
        #^the message of the post
    def save(self, *args, **kwargs):
        self.message_html = misaka.html(self.message)#if someone puts a link in their post, it doesn't conflict, it's supported in the html with Misaka
        super().save(*args, **kwargs)

    def get_absolute_url(self):
        return reverse(
            "posts:single",
            kwargs={
                "username": self.user.username,
                "pk": self.pk #primary keys are used as a way to relate posts back to a url
            }
        )

    class Meta:
        ordering = ["-created_at"] #most recent posts are at the top
        unique_together = ["user", "message"]#every message is uniquely linked to a user

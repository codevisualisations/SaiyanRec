#SOURCE: sections of code in this file are all taken from one source => https://www.udemy.com/course/python-and-django-full-stack-web-developer-bootcamp/learn/lecture/6649278?start=615#content
from django.contrib import messages
from django.contrib.auth.mixins import(
    LoginRequiredMixin,
    PermissionRequiredMixin
)
from django.urls import reverse
from django.db import IntegrityError
from django.shortcuts import get_object_or_404
from django.views import generic
from groups.models import Group,GroupMember #importing our groups
from . import models

class CreateGroup(LoginRequiredMixin, generic.CreateView): #if someones logged into the site, and they want to create their own group
    fields = ("name", "description") #create a name of group and a description
    model = Group #connect to group model

class SingleGroup(generic.DetailView):#details of the specific group e.g the posts inside that model
    model = Group

class ListGroups(generic.ListView):#a list of all the available groups
    model = Group


class JoinGroup(LoginRequiredMixin, generic.RedirectView):
#login required mixin-- you have to be logged in to join a group
    def get_redirect_url(self, *args, **kwargs):#grabs the url we want to redirect them too, if they join a group
        return reverse("groups:single",kwargs={"slug": self.kwargs.get("slug")})#once you join a group go back to that groups detail page

    def get(self, request, *args, **kwargs):
        group = get_object_or_404(Group,slug=self.kwargs.get("slug"))
        #warning message if a person is already inside a group
        try:
            GroupMember.objects.create(user=self.request.user,group=group)
            #try to get the group member objects, and create one where the current user, is
            #equal to the user, and group is equal to group
        except IntegrityError:
            messages.warning(self.request,("Warning, already a member of {}".format(group.name)))

        else:
            messages.success(self.request,"You are now a member of the {} group.".format(group.name))

        return super().get(request, *args, **kwargs)


class LeaveGroup(LoginRequiredMixin, generic.RedirectView):

    def get_redirect_url(self, *args, **kwargs):
        return reverse("groups:single",kwargs={"slug": self.kwargs.get("slug")})

    def get(self, request, *args, **kwargs):
        try:
            membership = models.GroupMember.objects.filter(
                user=self.request.user,
                group__slug=self.kwargs.get("slug")
            ).get()

        except models.GroupMember.DoesNotExist:
            messages.warning(
                self.request,
                "You can't leave this group because you aren't in it."
            )

        else:
            membership.delete()
            messages.success(
                self.request,
                "You have successfully left this group."
            )
        return super().get(request, *args, **kwargs)

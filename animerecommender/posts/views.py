#SOURCE: sections of code in this file are all taken from one source => https://www.udemy.com/course/python-and-django-full-stack-web-developer-bootcamp/learn/lecture/6649278?start=615#content
from django.contrib import messages
from django.contrib.auth.mixins import LoginRequiredMixin #someone needs to be logged in to perform certain actions
from django.urls import reverse_lazy
from django.http import Http404
from django.views import generic

from braces.views import SelectRelatedMixin

from . import forms
from . import models

from django.contrib.auth import get_user_model
User = get_user_model() #can call things off of this object


class PostList(SelectRelatedMixin, generic.ListView): #a list of posts belonging to a group
    model = models.Post
    select_related = ("user", "group") #allows a tuple of related models. The foreign keys for this post


class UserPosts(generic.ListView):
    model = models.Post
    template_name = "posts/user_post_list.html"

    def get_queryset(self):
        try:
            self.post_user = User.objects.prefetch_related("posts").get(
                username__iexact=self.kwargs.get("username")
            )#fetch posts from a related user
        except User.DoesNotExist:
            raise Http404
        else:
            return self.post_user.posts.all()

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["post_user"] = self.post_user
        return context #returning the context data object, related to the specific user who posted it


class PostDetail(SelectRelatedMixin, generic.DetailView):
    model = models.Post
    select_related = ("user", "group") #foreign keys

    def get_queryset(self):
        queryset = super().get_queryset()
        return queryset.filter(
            user__username__iexact=self.kwargs.get("username")
        )#here, get the query set for the post, and filter where the passed in username is
        #equal to the username, and it has to be exactly the user's username. The username off that models object


class CreatePost(LoginRequiredMixin, SelectRelatedMixin, generic.CreateView):
    # form_class = forms.PostForm
    fields = ('message','group')#if we want to create a post in a group, we put the message, and specify the group
    model = models.Post

    # def get_form_kwargs(self):
    #     kwargs = super().get_form_kwargs()
    #     kwargs.update({"user": self.request.user})
    #     return kwargs

    def form_valid(self, form):
        self.object = form.save(commit=False)
        self.object.user = self.request.user
        self.object.save()
        return super().form_valid(form)
        #connects the post to the user itself


class DeletePost(LoginRequiredMixin, SelectRelatedMixin, generic.DeleteView):
    model = models.Post
    select_related = ("user", "group")
    success_url = reverse_lazy("posts:all")
    #once a delete is confirmed,the user is returned to all the posts

    def get_queryset(self):
        queryset = super().get_queryset()
        return queryset.filter(user_id=self.request.user.id)

    def delete(self, *args, **kwargs):
        messages.success(self.request, "Post Deleted")
        return super().delete(*args, **kwargs)

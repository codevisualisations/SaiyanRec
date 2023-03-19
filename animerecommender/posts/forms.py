from django import forms
from . import models
#SOURCE: sections of code in this file are all taken from one source => https://www.udemy.com/course/python-and-django-full-stack-web-developer-bootcamp/learn/lecture/6649278?start=615#content

class PostForm(forms.ModelForm):
    class Meta:
        fields = ("message", "group")
        model = models.Post

    def __init__(self, *args, **kwargs):
        user = kwargs.pop("user", None)
        super().__init__(*args, **kwargs)
        if user is not None:
            self.fields["group"].queryset = (
                models.Group.objects.filter(
                    pk__in=user.groups.values_list("group__pk")
                )
            )

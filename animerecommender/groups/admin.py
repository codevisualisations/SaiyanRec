from django.contrib import admin
from . import models

# Register your models here.
class GroupMemberInline(admin.TabularInline):
    model = models.GroupMember

admin.site.register(models.Group)

#models are registered with the admin
#we can use a tabular inline class, so when I visit the admin page
#I can click on group, and see the group members, and edit them too

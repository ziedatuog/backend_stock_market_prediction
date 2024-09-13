from django.contrib import admin
from .models import *
# Register your models here.

# Custom Admin for CustomUser
class CustomUserAdmin(admin.ModelAdmin):
    list_display = ('email', 'username', 'is_staff', 'is_active')
    search_fields = ('email', 'username')
    list_filter = ('is_staff', 'is_active')
    ordering = ('email',)


admin.site.register(CustomUser)

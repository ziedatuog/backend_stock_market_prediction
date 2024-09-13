from django.contrib import admin
from .models import Company, StockData

# Username (leave blank to use 'zelalembirhan'): zelalem
# Email address: dezelalem31@gmail.com
# Password: 
# Password (again): 
# Superuser created successfully.

# Custom Admin for Company
class CompanyAdmin(admin.ModelAdmin):
    list_display = ('name', 'ticker')
    search_fields = ('name', 'ticker')
    ordering = ('ticker',)

# Custom Admin for StockData
class StockDataAdmin(admin.ModelAdmin):
    list_display = ('company', 'date', 'open', 'high', 'low', 'close', 'volume')
    search_fields = ('company__ticker', 'date')
    list_filter = ('company', 'date')
    ordering = ('-date',)

# Register your models here. 

admin.site.register(Company)
admin.site.register(StockData)
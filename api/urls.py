from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import CompanyViewSet, StockDataViewSet

router = DefaultRouter()
router.register('companies', CompanyViewSet, basename='company')
router.register('stockdata', StockDataViewSet, basename='stockdata')

urlpatterns = [
    path('', include(router.urls)),
     
]
 
# urlpatterns = [
#     path('stocks/', StockDataList.as_view(), name='stock_data_list'),
#     path('predict/<str:ticker>/', predict_next_day, name='predict_next_day'),
# ]
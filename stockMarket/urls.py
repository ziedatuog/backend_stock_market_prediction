 
from django.contrib import admin
from django.urls import path, include
from knox import views as knox_views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('api.urls')),
    path('custom_auth/', include('custom_auth.urls')), 
    path('api/auth/',include('knox.urls')),
    #*******************************8
    path('custom_auth/logout/', knox_views.LogoutView.as_view(), name='knox_logout'), 
    path('custom_auth/logoutall/', knox_views.LogoutAllView.as_view(), name='knox_logoutall'), 
    
    #********************************
    # path('logout/',knox_views.LogoutView.as_view(), name='knox_logout'), 
    # path('logoutall/',knox_views.LogoutAllView.as_view(), name='knox_logoutall'), 
    path('api/password_reset/',include('django_rest_passwordreset.urls', namespace='password_reset')), 
   
]


# from django.contrib import admin
# from django.urls import path, include
# from knox import views as knox_views

# urlpatterns = [
#     path('admin/', admin.site.urls),
#     path('',include('users.urls')), 
#     #path('api/auth/',include('knox.urls')), 

#     path('logout/',knox_views.LogoutView.as_view(), name='knox_logout'), 
#     path('logoutall/',knox_views.LogoutAllView.as_view(), name='knox_logoutall'), 
#     path('api/password_reset/',include('django_rest_passwordreset.urls', namespace='password_reset')), 

# ]
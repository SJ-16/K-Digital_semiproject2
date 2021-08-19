"""config URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include

from semiproject2 import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.index, name='index'),
    path('home2', views.home2, name='home2'),
    path('login', views.login, name='login'),
    path('loginimpl', views.loginimpl, name='loginimpl'),
    path('logout', views.logout, name='logout'),
    path('signup', views.signup, name='signup'),
    path('signupimpl', views.signupimpl, name='signupimpl'),
    path('recommend', views.recommend, name='recommend'),
    path('myinfo', views.myinfo, name='myinfo'),
    path('infoupdate', views.infoupdate, name='infoupdate'),
    path('infodelete', views.infodelete, name='infodelete'),

]

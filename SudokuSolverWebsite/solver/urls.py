from django.urls import path
from . import views

urlpatterns = [
    path('', views.upload_photo, name='upload_photo'),
    path('result/', views.result, name='result')
]

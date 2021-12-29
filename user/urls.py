from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='user'),
    path('request/', views.index, name='request_list'),
    path('request/create/', views.request_create, name='request_create'),
    path('request/<int:pk>/', views.request_detail, name='request_detail'),
]

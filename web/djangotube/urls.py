from django.urls import path
from . import views

app_name = 'video'

urlpatterns = [
    path('', views.video_list, name='list'),
    path('new/', views.add_video, name='new'),
    path('<int:video_id>/', views.detail_video, name='detail'),

]
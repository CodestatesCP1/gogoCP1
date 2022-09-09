from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.urls import reverse
from .models import Video

def video_list(request):
    video_list = Video.objects.all()
    context = {'video_list':video_list}
    return render(request, 'djangotube/video_list.html', context)

def add_video(request):
    if request.method == 'POST':
        title = request.POST['title']
        video_key = request.POST['video_key']
        Video.objects.create(title=title, video_key=video_key)
        return redirect(reverse('video:list'))
    elif request.method == 'GET':
        return render(request, 'djangotube/video_new.html')

def detail_video(request, video_id):
    video = Video.objects.get(id=video_id)
    return render(request, 'djangotube/video_detail.html', {'video':video})
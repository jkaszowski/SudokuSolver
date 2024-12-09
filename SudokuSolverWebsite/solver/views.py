import json

from django.shortcuts import render, redirect, reverse
from .forms import PhotoUploadForm

from .processing import process_photo

def upload_photo(request):
    if request.method == 'POST':
        form = PhotoUploadForm(request.POST, request.FILES)
        if form.is_valid():
            photo = form.cleaned_data['photo']
            is_yolo_needed = 'enableYolo' in request.POST

            # You can handle the uploaded file here
            with open(f'media/{photo.name}', 'wb+') as destination:
                for chunk in photo.chunks():
                    destination.write(chunk)
            ret, str = process_photo(f'media/{photo.name}')
            return redirect(f'result')

    else:
        form = PhotoUploadForm()

    return render(request, 'upload.html', {'form': form})


def result(request):
    photoid = request.GET.get('photo_id')
    original_url = f'/media/input.png'
    canny_url = f'/media/canny.png'
    final_url = f'/media/output.png'
    return render(request, 'result.html',{'original_url': original_url,'canny_url': canny_url,'final_url': final_url})
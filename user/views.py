from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
import requests
from django.conf import settings
from .forms import RequestForm, AssetForm
from django.contrib import messages
from django.contrib.auth.models import User
from . import models


@login_required
def index(request):
    template = 'dashboard/index.html'
    context = {}
    context['reqs'] = models.Request.objects.filter(user=request.user) 
    return render(request, template, context)


@login_required
def request_create(request):
    template = 'dashboard/request.html'
    context = {}
    if request.method == 'POST':
        form = RequestForm(request.POST)
        if form.is_valid():
            form = form.save(commit=False)
            form.user = request.user
            form.save()
            messages.success(request, 'Request created successfully')
            return redirect('request_detail', form.id)
        else:
            messages.error(request, 'Error creating request')
    else:
        form = RequestForm()

    context['form'] = form
    return render(request, template, context)


@login_required
def request_detail(request, pk):
    template = 'dashboard/request.html'
    context = {}
    request_obj = models.Request.objects.get(pk=pk)
    context['request_obj'] = request_obj
    if request.method == 'POST':
        form = AssetForm(request.POST, request.FILES)
        if form.is_valid():
            form = form.save(commit=False)
            form.request = request_obj
            form.save()
            messages.success(request, 'Asset created successfully')
            return redirect('request_detail', pk=pk)
        else:
            messages.error(request, 'Error creating asset')
    else:
        form = AssetForm()

    context['form'] = form
    context['assets'] = models.Asset.objects.filter(request=request_obj)
    context['request_detail'] = True
    return render(request, template, context)
from django.contrib.auth.models import User
from django.db import models
from django.db.models.signals import post_save
from django.dispatch import receiver
from django.utils import timezone
import datetime
import os
from user.classification import test


class Profile(models.Model):
    USER = 1
    DOCTOR = 2
    TYPE_CHOICES = (
        (USER, 'User'),
        (DOCTOR, 'Doctor'),
    )
     
    user            = models.OneToOneField(User, on_delete=models.CASCADE)
    image           = models.ImageField(upload_to='profile/%y/%m/%d/', null=True, blank=True)
    type            = models.IntegerField(choices=TYPE_CHOICES, default=1)
    
    updated         = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.user.username
        

@receiver(post_save, sender=User)
def create_or_update_user_profile(sender, instance, created, **kwargs):
    if created:
        Profile.objects.create(user=instance)
    instance.profile.save()


class Request(models.Model):

    STATUS_CHOICES = (
        (1, 'Created'),
        (2, 'Processed'),
        (3, 'Reviewed'),
    )

    user            = models.ForeignKey(User, on_delete=models.CASCADE)
    name            = models.CharField(max_length=100)
    age             = models.IntegerField(null=True, blank=True)
    doctor          = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True, related_name='doctor')
    status          = models.IntegerField(choices=STATUS_CHOICES, default=1)

    is_correct      = models.BooleanField(default=False)
    created         = models.DateTimeField(auto_now_add=True)
    updated         = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.user.username


class Asset(models.Model):
    
        request       = models.ForeignKey(Request, on_delete=models.CASCADE)
        file          = models.FileField(upload_to='asset/%y/%m/%d/')

        ignore        = models.BooleanField(default=False)
        created       = models.DateTimeField(auto_now_add=True)
        updated       = models.DateTimeField(auto_now=True)

        def __str__(self):
            return str(self.file.name) + ' - ' + str(self.request.user.username)


        def doctor_remark(self):
            try:
                return Remark.objects.filter(asset=self).first().doctor_remark
            except AttributeError:
                return None

        def system_remark(self):
            try:
                return Remark.objects.filter(asset=self).first().system_remark
            except AttributeError:
                return None


@receiver(post_save, sender=Asset)
def test_xray_and_update_remark(sender, instance, created, **kwargs):
    if created:
        path = instance.file.path
        print(path)
        status = test(path)
        if status > 0.5:
            Remark.objects.get_or_create(asset=instance, system_remark='Stone')
        else:
            Remark.objects.get_or_create(asset=instance, system_remark='Normal')



class Remark(models.Model):
    
        asset                = models.ForeignKey(Asset, on_delete=models.CASCADE)
        system_remark        = models.CharField(max_length=10, null=True, blank=True)
        doctor_remark        = models.CharField(max_length=10, null=True, blank=True)

        created       = models.DateTimeField(auto_now_add=True)
        updated       = models.DateTimeField(auto_now=True)

        def __str__(self):
            return str(self.asset.request.user.username)
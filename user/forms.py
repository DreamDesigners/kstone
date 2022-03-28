from allauth.account.forms import SignupForm
from django import forms
from . import models

class CustomSignupForm(SignupForm):
    first_name = forms.CharField(max_length=30, label='First Name')
    last_name = forms.CharField(max_length=30, label='Last Name')
    username = forms.CharField(label='Mobile Number')
    def signup(self, request, user):
        user.first_name = self.cleaned_data['first_name']
        user.last_name = self.cleaned_data['last_name']
        user.username = self.cleaned_data['username']
        user.save()
        return user


class RequestForm(forms.ModelForm):
    class Meta:
        model = models.Request
        fields = [
            'name',
            'age',
        ]
        widgets = {
            'name': forms.TextInput(attrs={'class': 'form-control'}),
            'age': forms.NumberInput(attrs={'class': 'form-control'}),
        }


class AssetForm(forms.ModelForm):
    class Meta:
        model = models.Asset
        fields = [
            'file',
        ]

        widgets = {
            'file': forms.FileInput(attrs={'class': 'form-control', 'required': 'required'}),
        }
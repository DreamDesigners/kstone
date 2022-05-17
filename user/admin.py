from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from django.contrib.auth.models import User

from .models import Profile, Request, Asset, Remark

admin.site.site_header = 'KidneyStone'
admin.site.site_title = 'Kidney Stone Detection'

class ProfileInline(admin.StackedInline):
    model = Profile
    can_delete = False
    verbose_name_plural = 'Profile'
    fk_name = 'user'


class CustomUserAdmin(UserAdmin):
    inlines = (ProfileInline,)
    list_display = ('username', 'email', 'first_name', 'last_name', 'is_staff')
    list_select_related = ('profile', )

    def get_inline_instances(self, request, obj=None):
        if not obj:
            return list()
        return super(CustomUserAdmin, self).get_inline_instances(request, obj)


admin.site.unregister(User)
admin.site.register(User, CustomUserAdmin)


class RemarkAdmin(admin.ModelAdmin):
    list_display = ('asset', 'system_remark', 'doctor_remark', 'created', 'updated')
    list_filter = ('system_remark', 'doctor_remark')

class RemarkInline(admin.StackedInline):
    model = Remark
    can_delete = False
    verbose_name_plural = 'Remark'
    fk_name = 'asset'
    extra = 1


class AssetItemAdmin(admin.ModelAdmin):
    list_display = ['request', 'file', 'created']
    inlines = [RemarkInline]


class AssetInLineAdmin(admin.TabularInline):
    model = Asset
    extra = 1


class RequestAdmin(admin.ModelAdmin):
    list_display = [
        'user',
        'name',
        'age',
        'doctor',
        'status',
        'created',
        'updated',
        ]
    inlines = [AssetInLineAdmin]


admin.site.register(Asset, AssetItemAdmin)
admin.site.register(Request, RequestAdmin)
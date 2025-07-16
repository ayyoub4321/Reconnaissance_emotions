from django.urls import path
from .views import index,display_image,display_modele,modelsExpli,resume,rapport
urlpatterns=[
    path('image/<int:image_id>/', display_image, name='charger_image'),
    path('',index,name='home'),
    path('model',display_modele,name='modelsML'),
    path('modelsExpli',modelsExpli,name='modelsExpli'),
    path('resume',resume,name='resume'),
    path('/rapport',rapport,name='rapport')
]
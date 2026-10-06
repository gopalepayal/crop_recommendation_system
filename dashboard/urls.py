from django.urls import path
from . import views

urlpatterns = [
    path('', views.landing, name='landing'),
    path('overview/', views.overview, name='overview'),
    path('crop-analysis/', views.crop_analysis, name='crop_analysis'),
    path('environmental-analysis/', views.environmental_analysis, name='environmental_analysis'),
    path('model-performance/', views.model_performance, name='model_performance'),
    path('recommendation/', views.crop_recommendation, name='crop_recommendation'),
    path('dataset-explorer/', views.dataset_explorer, name='dataset_explorer'),
]

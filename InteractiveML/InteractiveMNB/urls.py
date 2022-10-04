from django.urls import path
from . import views

urlpatterns = [
    path('', views.predictor, name="predictor"),
    path('add_word', views.add_word, name="addWord"),
    path('remove_word', views.rem_word, name="remWord"),
    path('adj_weight', views.adj_weight, name="adjWeight"),
]
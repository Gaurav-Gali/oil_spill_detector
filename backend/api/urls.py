from django.urls import path
# Views
from .views import (
    TestView
)
urlpatterns = [
    path("" , TestView.as_view() , name="TestView"),
]
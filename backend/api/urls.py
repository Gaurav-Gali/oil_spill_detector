from django.urls import path
# Views
from .views import (
    ShipData
)
urlpatterns = [
    path("" , ShipData.as_view() , name="TestView"),
]
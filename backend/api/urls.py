from django.urls import path
# Views
from .views import (
    ShipData
)
urlpatterns = [
    path("<str:limit>/" , ShipData.as_view() , name="TestView"),
]
# Rest Framework
from rest_framework import decorators
from rest_framework.response import Response
from rest_framework import status
from rest_framework.views import APIView

# Create your views here.
class TestView(APIView):
    def get(self , request):
        return Response({"message": "Hello World"}, status=status.HTTP_200_OK)
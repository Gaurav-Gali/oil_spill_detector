# Rest Framework
from rest_framework import decorators
from rest_framework.response import Response
from rest_framework import status
from rest_framework.views import APIView

import pandas as pd


# Create your views here.
class ShipData(APIView):
    def get(self , request):
        # Read the CSV file into a DataFrame
        df = pd.read_csv('https://raw.githubusercontent.com/abhinavanagarajan/election/main/oilspilldataset.csv')

# Convert the DataFrame to a list of dictionaries
        list_of_dicts = df.to_dict(orient='records')

# Print the list of dictionaries
#print(list_of_dicts)
        return Response(list_of_dicts, status=status.HTTP_200_OK)
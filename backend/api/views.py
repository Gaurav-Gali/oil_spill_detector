from rest_framework import decorators
from rest_framework.response import Response
from rest_framework import status
from rest_framework.views import APIView
import pandas as pd


# Create your views here.
class ShipData(APIView):
    def get(self, request, limit):
        # Get the 'limit' parameter from the request, default to None if not provided
        # limit = request.query_params.get('limit', None)
        # limit = int(limit)

        # Read the CSV file into a DataFrame
        df = pd.read_csv('https://raw.githubusercontent.com/abhinavanagarajan/election/main/oilspilldataset.csv')
        start,end = limit.split("-")
        # If a limit is provided, limit the number of records
        if limit:
            try:
                start = int(start)-1
                end = int(end)
                # limit = int(limit)
                # df = df.head(limit)
            except ValueError:
                return Response({"error": "Invalid limit value"}, status=status.HTTP_400_BAD_REQUEST)

        # Convert the DataFrame to a list of dictionaries
        list_of_dicts = df.to_dict(orient='records')[start:end]

        # Return the list of dictionaries with a status code 200
        return Response({"Data":list_of_dicts}, status=status.HTTP_200_OK)

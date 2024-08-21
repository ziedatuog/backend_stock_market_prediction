# stocks/views.py
from rest_framework import viewsets, permissions
from .models import Company, StockData
from .serializers import CompanySerializer, StockDataSerializer
from rest_framework.response import Response 
from rest_framework import status
from django.shortcuts import get_object_or_404
from rest_framework.decorators import action

class CompanyViewSet(viewsets.ModelViewSet):
    permission_classes = [permissions.AllowAny]
    queryset = Company.objects.all()
    serializer_class = CompanySerializer

class StockDataViewSet(viewsets.ModelViewSet):
    permission_classes = [permissions.AllowAny]
    queryset = StockData.objects.all()
    serializer_class = StockDataSerializer
    
    def list(self, request):
        queryset = self.get_queryset()
        serializer = self.serializer_class(queryset, many=True)
        return Response(serializer.data)
         

    def create(self, request):
        company_data = request.data.get('company')
        
        if not company_data:
            return Response({'detail': 'Company data is required.'}, status=status.HTTP_400_BAD_REQUEST)

        # Get or create the company based on the provided data
        company, created = Company.objects.get_or_create(
            name=company_data.get('name'),
            ticker=company_data.get('ticker')
        )
        
        # Assign the company ID to the request data
        request.data['company'] = company.id
        
        # Proceed with the StockData creation
        serializer = self.serializer_class(data=request.data)
        
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
   
    
    def retrieve(self, request, pk=None):
        # Get the specific company based on primary key (pk)
        company = get_object_or_404(Company, pk=pk)
        
        # Get all StockData associated with this company
        stock_data = StockData.objects.filter(company=company)
        
        # Serialize the company data
        company_serializer = self.serializer_class(company)
        
        # Serialize the stock data
        stock_data_serializer = StockDataSerializer(stock_data, many=True)
        
        # Combine the serialized company data with its associated stock data
        response_data = company_serializer.data
        response_data['stock_data'] = stock_data_serializer.data
        
        # Return the combined data as the response
        return Response(response_data, status=status.HTTP_200_OK)
    #*******************************8
        

    def update(self, request, pk=None):
        # Get the specific StockData instance
        stock_data = get_object_or_404(StockData, pk=pk)

        # Extract and update the company information if provided
        company_data = request.data.get('company')
        if company_data:
            # Update or create the company based on the provided data
            company, created = Company.objects.get_or_create(
                name=company_data.get('name'),
                ticker=company_data.get('ticker')
            )
            # Assign the company ID to the request data
            request.data['company'] = company.id
        
        # Proceed with updating the StockData instance
        serializer = self.serializer_class(stock_data, data=request.data)
        
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_200_OK)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    def destroy(self, request, pk=None):
        company = get_object_or_404(Company, pk=pk)
        company.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)
         
    
    
    

from rest_framework import generics
from rest_framework.response import Response
from rest_framework.decorators import api_view
from .models import Company, StockData
from .serializers import StockDataSerializer
from django.shortcuts import get_object_or_404
import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

#7777777777777777777777777777777777777777777777777777777777777777777 past prediction function

# class StockDataList(generics.ListAPIView):
#     serializer_class = StockDataSerializer

#     def get_queryset(self):
#         ticker = self.request.query_params.get('ticker', None)
#         start_date = self.request.query_params.get('start_date', None)
#         end_date = self.request.query_params.get('end_date', datetime.date.today())
        
#         if ticker and start_date:
#             company = get_object_or_404(Company, ticker=ticker)
#             queryset = StockData.objects.filter(
#                 company=company,
#                 date__range=[start_date, end_date]
#             )
#             return queryset
#         return StockData.objects.none()

# @api_view(['GET'])
# def predict_next_day(request, ticker):
#     company = get_object_or_404(Company, ticker=ticker)

#     # Fetch historical data from the database
#     stock_data = StockData.objects.filter(company=company).order_by('-date')
    
#     if stock_data.count() < 100:
#         return Response({"error": "Not enough data to make a prediction. At least 100 days of data are required."})

#     # Prepare the data for prediction
#     data = pd.DataFrame(list(stock_data.values('date', 'close')))
#     data.set_index('date', inplace=True)
#     data = data.sort_index()

#     data_training = data['close'][:int(len(data) * 0.70)]
#     data_testing = data['close'][int(len(data) * 0.70):]

#     scaler = MinMaxScaler(feature_range=(0, 1))
#     data_training_array = scaler.fit_transform(data_training.values.reshape(-1, 1))

#     # Load model
#     model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Model', 'stock_prediction_model.pkl')
#     model = joblib.load(model_path)

#     # Prepare test data
#     past_100_days = data_training.tail(100).values
#     final_df = np.concatenate([past_100_days, data_testing.values])
#     input_data = scaler.transform(final_df.reshape(-1, 1))

#     x_test = []
#     for i in range(100, input_data.shape[0]):
#         x_test.append(input_data[i - 100:i, 0])

#     x_test = np.array(x_test)
#     x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

#     # Make prediction
#     y_predicted = model.predict(x_test[-1].reshape(1, 100, 1))
#     predicted_close = y_predicted[0][0]
#     predicted_close = predicted_close * (1 / scaler.scale_[0])

#     prediction = {
#         "ticker": ticker,
#         "predicted_date": (data.index[-1] + pd.Timedelta(days=1)).isoformat(),
#         "predicted_close": predicted_close
#     }
#     return Response(prediction)

#777777777777777777777777777777777777777777777777777777777777777777777

def predict_next_30_days(self, request, pk=None):
        company = get_object_or_404(Company, pk=pk)
        stock_data = StockData.objects.filter(company=company).order_by('-date')

        if stock_data.count() < 50:
            return Response({"error": "Not enough data to make a prediction. At least 50 days of data are required."}, status=status.HTTP_400_BAD_REQUEST)

        data = pd.DataFrame(list(stock_data.values('date', 'close')))
        data.set_index('date', inplace=True)
        data = data.sort_index()

        data_training = data['close'][:int(len(data) * 0.70)]
        data_testing = data['close'][int(len(data) * 0.70):]

        scaler = MinMaxScaler(feature_range=(0, 1))
        data_training_array = scaler.fit_transform(data_training.values.reshape(-1, 1))

        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model', 'stock_prediction_model.pkl')
        model = joblib.load(model_path)

        past_50_days = data_training.tail(50).values
        final_df = np.concatenate([past_50_days, data_testing.values])
        input_data = scaler.transform(final_df.reshape(-1, 1))

        x_test = []
        for i in range(50, input_data.shape[0]):
            x_test.append(input_data[i - 50:i, 0])

        x_test = np.array(x_test)
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

        y_predicted = model.predict(x_test)
        y_predicted = y_predicted * (1 / scaler.scale_[0])

        # Predict the next 30 days
        last_50_days = input_data[-50:]
        future_predictions = []
        for _ in range(30):
            pred = model.predict(last_50_days.reshape(1, 50, 1))
            future_predictions.append(pred[0][0])
            last_50_days = np.append(last_50_days[1:], pred)

        future_predictions = np.array(future_predictions)
        future_predictions = future_predictions * (1 / scaler.scale_[0])

        future_dates = pd.date_range(start=data.index[-1] + timedelta(days=1), periods=30)

        # Create a DataFrame for future predictions
        future_df = pd.DataFrame(data=future_predictions, index=future_dates, columns=['close'])

        # Concatenate the test data predictions with future predictions
        predicted_df = pd.DataFrame(data=y_predicted, index=data_testing.index, columns=['close'])
        result_df = pd.concat([data_training, predicted_df, future_df])

        # Plotting the results
        plt.figure(figsize=(14, 5))
        plt.plot(result_df.index, result_df['close'], color='blue', label='Predicted')
        plt.plot(data_testing.index, data_testing, color='red', label='Actual (Test Data)')
        plt.xlabel('Date')
        plt.ylabel('Stock Price')
        plt.title(f'Stock Price Prediction for {company.ticker}')
        plt.legend()

        # Convert the plot to a PNG image
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()

        return Response({
            "ticker": company.ticker,
            "predicted_data": result_df.to_dict(),
            "plot": image_base64,
        }, status=status.HTTP_200_OK)
        
        
        
        
        
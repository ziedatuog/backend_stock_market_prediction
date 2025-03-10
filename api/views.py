# stocks/views.py
 
from rest_framework import viewsets, permissions
from .models import Company, StockData
from .serializers import CompanySerializer, StockDataSerializer
from rest_framework.response import Response
from rest_framework import status
from django.shortcuts import get_object_or_404
from rest_framework.decorators import action
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
import matplotlib 
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # Use Agg backend for non-GUI rendering

import base64
from io import BytesIO
from datetime import timedelta

class CompanyViewSet(viewsets.ModelViewSet):
    permission_classes = [permissions.AllowAny]
    queryset = Company.objects.all()
    serializer_class = CompanySerializer
    
    def create(self, request):
        serializer = self.serializer_class(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
    
    
 
    
     
    @action(detail=True, methods=['get'], url_path='predict_next_30_days')
    def predict_next_30_days(self, request, pk=None):
        company = get_object_or_404(Company, pk=pk)
        stock_data = StockData.objects.filter(company=company).order_by('-date')

        if stock_data.count() < 50:
            return Response({"error": "Not enough data to make a prediction. At least 50 days of data are required."}, status=status.HTTP_400_BAD_REQUEST)

        data = pd.DataFrame(list(stock_data.values('date', 'close')))
        data['date'] = pd.to_datetime(data['date'])
        data.set_index('date', inplace=True)
        data = data.sort_index()

        # Split data into training and testing
        data_training = data['close'][:int(len(data) * 0.70)]
        data_testing = data['close'][int(len(data) * 0.70):]

        scaler = MinMaxScaler(feature_range=(0, 1))
        data_training_array = scaler.fit_transform(data_training.values.reshape(-1, 1))

        # Load the pre-trained model
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Model', 'futur_stock_prediction_model.pkl')
        model = joblib.load(model_path)

        # Prepare input data for predictions
        past_50_days = data_training.tail(50).values
        final_df = np.concatenate([past_50_days, data_testing.values])
        input_data = scaler.transform(final_df.reshape(-1, 1))

        x_test = [input_data[i - 50:i, 0] for i in range(50, input_data.shape[0])]
        x_test = np.array(x_test).reshape(-1, 50, 1)

        # Predict the test data
        y_predicted = model.predict(x_test)
        y_predicted = scaler.inverse_transform(y_predicted.reshape(-1, 1)).flatten()

        # Predict the next 30 days
        last_50_days = input_data[-50:]
        future_predictions = []
        for _ in range(30):
            pred = model.predict(last_50_days.reshape(1, 50, 1))
            future_predictions.append(pred[0][0])
            last_50_days = np.append(last_50_days[1:], pred)

        future_predictions = np.array(future_predictions).reshape(-1, 1)
        future_predictions = scaler.inverse_transform(future_predictions).flatten()

        # Create DataFrame for future predictions
        future_dates = pd.date_range(start=data.index[-1] + timedelta(days=1), periods=30)
        future_df = pd.DataFrame(data=future_predictions, index=future_dates, columns=['close'])

        # Create DataFrame for predicted test data
        predicted_df_index = data_testing.index[:len(y_predicted)]
        predicted_df = pd.DataFrame(data=y_predicted, index=predicted_df_index, columns=['close'])

        # Concatenate data and predictions
        result_df = pd.concat([data_training, predicted_df, future_df])

        # Convert dates to strings for JSON serialization
        result_df.index = result_df.index.strftime('%Y-%m-%d')
        future_df.index = future_df.index.strftime('%Y-%m-%d')

        # Plotting the results
        plt.figure(figsize=(14, 5))
        plt.plot(result_df.index, result_df['close'], color='blue', label='Predicted', linewidth=2)

        # Plot the future predictions
        plt.plot(future_df.index, future_df['close'], color='green', linestyle='--', label='Future Predictions')

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

        # Convert future_df to a dictionary for the table
        future_predictions_table = future_df.reset_index().rename(columns={'index': 'date', 'close': 'predicted_close'}).to_dict(orient='records')

        return Response({
            "ticker": company.ticker,
            "predicted_data": result_df.to_dict(),
            "future_predictions_table": future_predictions_table,  # Add table data here
            "plot": image_base64,
        }, status=status.HTTP_200_OK) 
    
class StockDataViewSet(viewsets.ModelViewSet):
    permission_classes = [permissions.AllowAny]
    queryset = StockData.objects.all()
    serializer_class = StockDataSerializer
    
    @action(detail=True, methods=['get'], url_path='stockdata')
    def get_stock_data(self, request, pk=None):
        company = get_object_or_404(Company, pk=pk)
        stock_data = StockData.objects.filter(company=company)
        stock_data_serializer = StockDataSerializer(stock_data, many=True)
        return Response(stock_data_serializer.data, status=status.HTTP_200_OK)
    
    def list(self, request):
        queryset = self.get_queryset()
        serializer = self.serializer_class(queryset, many=True)
        return Response(serializer.data)
         

    def create(self, request):
        company_id = request.data.get('company')

        if not company_id:
            return Response({'detail': 'Company ID is required.'}, status=status.HTTP_400_BAD_REQUEST)

        # Get the company object
        company = get_object_or_404(Company, id=company_id)

        # Assign the company object to the request data
        request.data['company'] = company.id

        # Proceed with the StockData creation
        serializer = self.serializer_class(data=request.data)
        
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    
     
    def retrieve(self, request, pk=None):
        # Get the specific Company instance
        company = get_object_or_404(Company, pk=pk)
        
        # Get the related StockData for this company
        stock_data = StockData.objects.filter(company=company)
        
        # Serialize the company and stock data
        company_serializer = CompanySerializer(company)
        stock_data_serializer = StockDataSerializer(stock_data, many=True)
        
        # Combine the serialized data into a single response
        response_data = company_serializer.data
        response_data['stock_data'] = stock_data_serializer.data
        
        return Response(response_data, status=status.HTTP_200_OK)
     
    
    
    def update(self, request, pk=None):
        # Get the specific StockData instance
        stock_data = get_object_or_404(StockData, pk=pk)
        
        # Extract and update the company information if provided
        company_data = request.data.get('company')
        if company_data:
            # Update or create the company based on the provided data
            company, created = Company.objects.update_or_create(
                ticker=company_data.get('ticker'),
                defaults={'name': company_data.get('name')}
            )
            # Assign the company ID to the request data
            request.data['company'] = company.id
        
        # Proceed with updating the StockData instance
        serializer = StockDataSerializer(stock_data, data=request.data)
        
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_200_OK)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
     
        
 
    
    def destroy(self, request, pk=None):
        company = get_object_or_404(Company, pk=pk)
        company.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)
    
    
     
    
   

         
   
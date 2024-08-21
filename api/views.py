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
import matplotlib.pyplot as plt
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
        
   
    
    ####***************************************
    
    @action(detail=True, methods=['get'], url_path='predict_next_30_days')
    def predict_next_30_days(self, request, pk=None):
        company = get_object_or_404(Company, pk=pk)
        stock_data = StockData.objects.filter(company=company).order_by('-date')

        if stock_data.count() < 50:
            return Response({"error": "Not enough data to make a prediction. At least 50 days of data are required."}, status=status.HTTP_400_BAD_REQUEST)

        data = pd.DataFrame(list(stock_data.values('date', 'close')))
        data['date'] = pd.to_datetime(data['date'])  # Ensure 'date' is in datetime format
        data.set_index('date', inplace=True)
        data = data.sort_index()

        data_training = data['close'][:int(len(data) * 0.70)]
        data_testing = data['close'][int(len(data) * 0.70):]

        scaler = MinMaxScaler(feature_range=(0, 1))
        data_training_array = scaler.fit_transform(data_training.values.reshape(-1, 1))

        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Model', 'futur_stock_prediction_model.pkl')
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

        # Create DataFrame for future predictions
        future_df = pd.DataFrame(data=future_predictions, index=future_dates, columns=['close'])

        # Ensure the predicted_df has the correct index length
        if len(y_predicted) != len(data_testing):
            predicted_df = pd.DataFrame(data=y_predicted, index=data_testing.index[:len(y_predicted)], columns=['close'])
        else:
            predicted_df = pd.DataFrame(data=y_predicted, index=data_testing.index, columns=['close'])

        # Concatenate the data and predictions
        result_df = pd.concat([data_training, predicted_df, future_df])

        # Convert dates to strings for JSON serialization
        result_df.index = result_df.index.strftime('%Y-%m-%d')
        future_df.index = future_df.index.strftime('%Y-%m-%d')

        # Plotting the results
        plt.figure(figsize=(14, 5))
        plt.plot(result_df.index, result_df['close'], color='blue', label='Predicted')
        plt.plot(data_testing.index.strftime('%Y-%m-%d'), data_testing, color='red', label='Actual (Test Data)')
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
    
    #**********************************

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
    
    
    # @action(detail=True, methods=['get'], url_path='predict_next_30_days')
    # def predict_next_30_days(self, request, pk=None):
    #     company = get_object_or_404(Company, pk=pk)
    #     stock_data = StockData.objects.filter(company=company).order_by('-date')

    #     if stock_data.count() < 50:
    #         return Response({"error": "Not enough data to make a prediction. At least 50 days of data are required."}, status=status.HTTP_400_BAD_REQUEST)

    #     data = pd.DataFrame(list(stock_data.values('date', 'close')))
    #     data.set_index('date', inplace=True)
    #     data = data.sort_index()

    #     data_training = data['close'][:int(len(data) * 0.70)]
    #     data_testing = data['close'][int(len(data) * 0.70):]

    #     scaler = MinMaxScaler(feature_range=(0, 1))
    #     data_training_array = scaler.fit_transform(data_training.values.reshape(-1, 1))

    #     model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Model', 'futur_stock_prediction_model.pkl')
    #     model = joblib.load(model_path)

    #     past_50_days = data_training.tail(50).values
    #     final_df = np.concatenate([past_50_days, data_testing.values])
    #     input_data = scaler.transform(final_df.reshape(-1, 1))

    #     x_test = []
    #     for i in range(50, input_data.shape[0]):
    #         x_test.append(input_data[i - 50:i, 0])

    #     x_test = np.array(x_test)
    #     x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

    #     y_predicted = model.predict(x_test)
    #     y_predicted = y_predicted * (1 / scaler.scale_[0])

    #     # Predict the next 30 days
    #     last_50_days = input_data[-50:]
    #     future_predictions = []
    #     for _ in range(30):
    #         pred = model.predict(last_50_days.reshape(1, 50, 1))
    #         future_predictions.append(pred[0][0])
    #         last_50_days = np.append(last_50_days[1:], pred)

    #     future_predictions = np.array(future_predictions)
    #     future_predictions = future_predictions * (1 / scaler.scale_[0])

    #     future_dates = pd.date_range(start=data.index[-1] + timedelta(days=1), periods=30)

    #     # Create a DataFrame for future predictions
    #     future_df = pd.DataFrame(data=future_predictions, index=future_dates, columns=['close'])

    #     # Concatenate the test data predictions with future predictions
    #     predicted_df = pd.DataFrame(data=y_predicted, index=data_testing.index, columns=['close'])
    #     result_df = pd.concat([data_training, predicted_df, future_df])

    #     # Plotting the results
    #     plt.figure(figsize=(14, 5))
    #     plt.plot(result_df.index, result_df['close'], color='blue', label='Predicted')
    #     plt.plot(data_testing.index, data_testing, color='red', label='Actual (Test Data)')
    #     plt.xlabel('Date')
    #     plt.ylabel('Stock Price')
    #     plt.title(f'Stock Price Prediction for {company.ticker}')
    #     plt.legend()

    #     # Convert the plot to a PNG image
    #     buf = BytesIO()
    #     plt.savefig(buf, format='png')
    #     buf.seek(0)
    #     image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    #     buf.close()

    #     return Response({
    #         "ticker": company.ticker,
    #         "predicted_data": result_df.to_dict(),
    #         "plot": image_base64,
    #     }, status=status.HTTP_200_OK)

         
    ################################################### 
    # def create(self, request):
    #     company_ticker = request.data.get('company')
    #     date_str = request.data.get('date')

    #     # Debug logging to verify received data
    #     print(f"Received data: {request.data}")

    #     if not company_ticker:
    #         return Response({'detail': 'Company ticker is required.'}, status=status.HTTP_400_BAD_REQUEST)
    #     if not date_str:
    #         return Response({'detail': 'Date is required.'}, status=status.HTTP_400_BAD_REQUEST)

    #     # Retrieve the company based on ticker
    #     company = get_object_or_404(Company, ticker=company_ticker)
        
    #     # Convert date string to a date object
    #     try:
    #         date = datetime.strptime(date_str, '%Y-%m-%d').date()
    #     except ValueError:
    #         return Response({'detail': 'Invalid date format. Use YYYY-MM-DD.'}, status=status.HTTP_400_BAD_REQUEST)
        
    #     # Check if stock data for this date already exists for the given company
    #     if StockData.objects.filter(company=company, date=date).exists():
    #         return Response({'detail': 'Stock data for this date already exists for the given company.'}, 
    #                         status=status.HTTP_400_BAD_REQUEST)
        
    #     # Make the QueryDict mutable so we can modify it
    #     request.data._mutable = True
    #     # Replace the company field with the company id
    #     request.data['company'] = company.id
    #     # Make the QueryDict immutable again
    #     request.data._mutable = False

    #     # Serialize and save the stock data
    #     serializer = self.serializer_class(data=request.data)
    #     if serializer.is_valid():
    #         serializer.save()
    #         return Response(serializer.data, status=status.HTTP_201_CREATED)
    #     else:
    #         print(f"Serializer errors: {serializer.errors}")  # Print serializer errors for debugging
    #         return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    ###############################################
             
    # def retrieve(self, request, pk=None):
    #     # Get the specific company based on primary key (pk)
    #     company = get_object_or_404(Company, pk=pk)
        
    #     # Get all StockData associated with this company
    #     stock_data = StockData.objects.filter(company=company)
        
    #     # Serialize the company data
    #     company_serializer = self.serializer_class(company)
        
    #     # Serialize the stock data
    #     stock_data_serializer = StockDataSerializer(stock_data, many=True)
        
    #     # Combine the serialized company data with its associated stock data
    #     response_data = company_serializer.data
    #     response_data['stock_data'] = stock_data_serializer.data
        
    #     return Response(response_data)
    #*******************************8
    
    

# from rest_framework import generics
# from rest_framework.response import Response
# from rest_framework.decorators import api_view
# from .models import Company, StockData
# from .serializers import StockDataSerializer
# from django.shortcuts import get_object_or_404
# import datetime
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
# import joblib
# import os

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

# def predict_next_30_days(self, request, pk=None):
#         company = get_object_or_404(Company, pk=pk)
#         stock_data = StockData.objects.filter(company=company).order_by('-date')

#         if stock_data.count() < 50:
#             return Response({"error": "Not enough data to make a prediction. At least 50 days of data are required."}, status=status.HTTP_400_BAD_REQUEST)

#         data = pd.DataFrame(list(stock_data.values('date', 'close')))
#         data.set_index('date', inplace=True)
#         data = data.sort_index()

#         data_training = data['close'][:int(len(data) * 0.70)]
#         data_testing = data['close'][int(len(data) * 0.70):]

#         scaler = MinMaxScaler(feature_range=(0, 1))
#         data_training_array = scaler.fit_transform(data_training.values.reshape(-1, 1))

#         model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model', 'stock_prediction_model.pkl')
#         model = joblib.load(model_path)

#         past_50_days = data_training.tail(50).values
#         final_df = np.concatenate([past_50_days, data_testing.values])
#         input_data = scaler.transform(final_df.reshape(-1, 1))

#         x_test = []
#         for i in range(50, input_data.shape[0]):
#             x_test.append(input_data[i - 50:i, 0])

#         x_test = np.array(x_test)
#         x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

#         y_predicted = model.predict(x_test)
#         y_predicted = y_predicted * (1 / scaler.scale_[0])

#         # Predict the next 30 days
#         last_50_days = input_data[-50:]
#         future_predictions = []
#         for _ in range(30):
#             pred = model.predict(last_50_days.reshape(1, 50, 1))
#             future_predictions.append(pred[0][0])
#             last_50_days = np.append(last_50_days[1:], pred)

#         future_predictions = np.array(future_predictions)
#         future_predictions = future_predictions * (1 / scaler.scale_[0])

#         future_dates = pd.date_range(start=data.index[-1] + timedelta(days=1), periods=30)

#         # Create a DataFrame for future predictions
#         future_df = pd.DataFrame(data=future_predictions, index=future_dates, columns=['close'])

#         # Concatenate the test data predictions with future predictions
#         predicted_df = pd.DataFrame(data=y_predicted, index=data_testing.index, columns=['close'])
#         result_df = pd.concat([data_training, predicted_df, future_df])

#         # Plotting the results
#         plt.figure(figsize=(14, 5))
#         plt.plot(result_df.index, result_df['close'], color='blue', label='Predicted')
#         plt.plot(data_testing.index, data_testing, color='red', label='Actual (Test Data)')
#         plt.xlabel('Date')
#         plt.ylabel('Stock Price')
#         plt.title(f'Stock Price Prediction for {company.ticker}')
#         plt.legend()

#         # Convert the plot to a PNG image
#         buf = BytesIO()
#         plt.savefig(buf, format='png')
#         buf.seek(0)
#         image_base64 = base64.b64encode(buf.read()).decode('utf-8')
#         buf.close()

#         return Response({
#             "ticker": company.ticker,
#             "predicted_data": result_df.to_dict(),
#             "plot": image_base64,
#         }, status=status.HTTP_200_OK)
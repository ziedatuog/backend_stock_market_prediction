# api/serializers.py

from rest_framework import serializers
from .models import Company, StockData
 
 

class StockDataSerializer(serializers.ModelSerializer):
    
    company = serializers.PrimaryKeyRelatedField(queryset=Company.objects.all())
    id = serializers.IntegerField(read_only=True)  # Explicitly add the `id` field

    class Meta:
        model = StockData
        fields = '__all__'
    
    def to_representation(self, instance):
        # print(instance)  # Debugging line
        representation = super().to_representation(instance)
        company_representation = {
            'id': instance.company.id,
            'name': instance.company.name,
            'ticker': instance.company.ticker
        }
        representation['company'] = company_representation
        return representation
    

class CompanySerializer(serializers.ModelSerializer):
    stock_data = StockDataSerializer(many=True, read_only=True)
    class Meta:
        model = Company
        # fields = '__all__'
        fields = ['id', 'name', 'ticker', 'stock_data']



    
############################

# class StockDataSerializer(serializers.ModelSerializer):
#     company = serializers.SlugRelatedField(
#         slug_field='ticker',  # Use the 'ticker' field instead of the ID
#         queryset=Company.objects.all()
#     )

#     class Meta:
#         model = StockData
#         fields = '__all__'

#############################


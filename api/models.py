from django.db import models

class Company(models.Model):
    name = models.CharField(max_length=100)
    ticker = models.CharField(max_length=10, unique=True)

    def __str__(self):
           return self.ticker
         
         

class StockData(models.Model):
    company = models.ForeignKey(Company, on_delete=models.CASCADE, related_name='stock_data')
    date = models.DateField()
    open = models.FloatField()
    high = models.FloatField()
    low = models.FloatField()
    close = models.FloatField()
    volume = models.IntegerField()

    class Meta:
        unique_together = ('company', 'date')

    def __str__(self):
        return f"{self.company.ticker} - {self.date}"
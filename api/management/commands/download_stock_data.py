from django.core.management.base import BaseCommand
import yfinance as yf
from api.models import Company, StockData

class Command(BaseCommand):
    help = 'Download and save stock data from Yahoo Finance'

    def add_arguments(self, parser):
        parser.add_argument('ticker', type=str, help='Stock ticker symbol to download data for')

    def handle(self, *args, **kwargs):
        ticker = kwargs['ticker']
        self.download_and_save_stock_data(ticker)

    def download_and_save_stock_data(self, ticker):
        self.stdout.write(f'Starting download for ticker: {ticker}')
        
        # Check if the company already exists in the database
        company, created = Company.objects.get_or_create(ticker=ticker, defaults={'name': ticker})
        
        try:
            # Download stock data from Yahoo Finance (last 3 months)
            data = yf.download(ticker, period="3mo")
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Error downloading data: {e}'))
            return

        if data.empty:
            self.stdout.write(self.style.ERROR(f'No data found for ticker {ticker}'))
            return

        # Loop over the rows in the dataframe and save them to the database
        for index, row in data.iterrows():
            try:
                StockData.objects.update_or_create(
                    company=company,
                    date=index.date(),
                    defaults={
                        'open': row['Open'],
                        'high': row['High'],
                        'low': row['Low'],
                        'close': row['Close'],
                        'volume': int(row['Volume'])
                    }
                )
            except Exception as e:
                self.stdout.write(self.style.ERROR(f'Error saving data for {index.date()}: {e}'))

        self.stdout.write(self.style.SUCCESS(f'Successfully added data for {ticker}'))

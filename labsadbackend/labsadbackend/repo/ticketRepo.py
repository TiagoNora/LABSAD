import pymongo
from pymongo import MongoClient, InsertOne
from datetime import datetime
import yfinance as yf
from datetime import datetime, timedelta

uri = 'mongodb://mongoadmin:a79c987b4dce244e9bc21620@vsgate-s1.dei.isep.ipp.pt:10777'  # MongoDB connection URI
db_name = 'labsad'       # Replace with your database name
collection_name = 'stockPrices'  # Replace with your collection name

client = MongoClient(uri)
db = client[db_name]
collection = db[collection_name]

class TicketRepo:
    def __init__(self):
        pass
    async def searchValueOfTicketFromDate(self, symbol, date):
        target_date = datetime.strptime(date, "%Y-%m-%d")
    
        gte = target_date
        lt = target_date + timedelta(days=1)
        ticker = yf.Ticker(symbol)
        tickerPrices = ticker.history(start=gte, end=lt)
        
        tickerPrices.reset_index(inplace=True)
        return tickerPrices.to_dict(orient='records')

import pymongo
from datetime import datetime
import yfinance as yf
from datetime import datetime, timedelta

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

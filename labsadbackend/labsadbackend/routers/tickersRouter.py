from fastapi import APIRouter, Depends
import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup as bs
import os

router = APIRouter(prefix='/tickers', tags=['TICKERS'])

@router.get('/', summary="Get all tickers")
async def getTickers():
    return {"message": "Hello World"}

@router.get('/updateTickers', summary="Update all tickers")
async def updateTickers():
    url = 'https://www.slickcharts.com/sp500'

    request = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
    soup = bs(request.text, "lxml")
    stats = soup.find('table', class_='table table-hover table-borderless table-sm')
    df = pd.read_html(str(stats))[0]
    df['% Chg'] = df['% Chg'].str.strip('()-%')
    df['% Chg'] = pd.to_numeric(df['% Chg'])
    df['Chg'] = pd.to_numeric(df['Chg'])

    save_directory = "../files"
    os.makedirs(save_directory, exist_ok=True)

    for symbol in df['Symbol']:
        ticker = yf.Ticker(symbol)
        tickerPrices = ticker.history(period="max", interval="1d")
        filename = os.path.join(save_directory, f"{symbol}_daily_data.csv")
        tickerPrices.to_csv(filename, index=False)
        print(f"Saved data for {symbol} to {filename}")
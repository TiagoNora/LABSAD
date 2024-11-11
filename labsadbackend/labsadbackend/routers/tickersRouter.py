from fastapi import APIRouter, Depends
import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup as bs
import os

router = APIRouter(prefix='/tickers', tags=['TICKERS'])

@router.get('/historialData', summary="Get ticker data")
async def getTicker(symbol: str):
    ticker = yf.Ticker(symbol)
    tickerPrices = ticker.history(period="max")
    tickerPrices.reset_index(inplace=True)
    return tickerPrices.to_dict(orient='records')

@router.get('/', summary="Get all tickers")
async def getTickers():
    url = 'https://www.slickcharts.com/sp500'

    request = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
    soup = bs(request.text, "lxml")
    stats = soup.find('table', class_='table table-hover table-borderless table-sm')
    df = pd.read_html(str(stats))[0]

    
    
    df['% Chg'] = df['% Chg'].str.strip('()-%')
    df['% Chg'] = pd.to_numeric(df['% Chg'], errors='coerce')
    df['Chg'] = pd.to_numeric(df['Chg'], errors='coerce')
    
    #df = df.drop(columns=['Chg', '% Chg'])
    df = df.fillna(0)
    
    df['Image'] = df['Symbol'].apply(lambda symbol: f'https://assets.parqet.com/logos/symbol/{symbol}?format=png')


    #print(df)
    #print(df.isna().sum())
    
    return df.to_dict(orient='records')

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
        tickerPrices = ticker.history(period="max")
        print(tickerPrices)
        filename = os.path.join(save_directory, f"{symbol}_daily_data.csv")
        tickerPrices.to_csv(filename)
        print(f"Saved data for {symbol} to {filename}")
from fastapi import APIRouter, Depends, HTTPException
import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup as bs
import os
import json
from datetime import datetime, timedelta
from labsadbackend.repo import *

router = APIRouter(prefix='/tickers', tags=['TICKERS'])

def load_sp500_tickers():
    try:
        with open('companysInfo.json', 'r') as file:
            return json.load(file)
        return tickers
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Ticker file not found")
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Error decoding the ticker file")
    
tickers = load_sp500_tickers()


@router.get('/search', summary="Search for a ticker")
async def searchTicker(name: str):
    matching_tickers = [company for company in tickers if company["Company"].lower().startswith(name.lower())]

    if not matching_tickers:
        raise HTTPException(status_code=404, detail="No companies found for the provided ticker")

    # Return a list of matching tickers and names
    return matching_tickers

@router.get('/historialData', summary="Get ticker data")
async def getTicker(symbol: str):
    ticker = yf.Ticker(symbol)
    tickerPrices = ticker.history(period="5y")
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

@router.get('/reducedInfo', summary="Get all tickers with reduced info")
async def getTickers():
    url = 'https://www.slickcharts.com/sp500'

    request = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
    soup = bs(request.text, "lxml")
    stats = soup.find('table', class_='table table-hover table-borderless table-sm')
    df = pd.read_html(str(stats))[0]
    
    df['% Chg'] = df['% Chg'].str.strip('()-%')
    df['% Chg'] = pd.to_numeric(df['% Chg'])
    df['Chg'] = pd.to_numeric(df['Chg'])
    
    df['Image'] = df['Symbol'].apply(lambda symbol: f'https://assets.parqet.com/logos/symbol/{symbol}?format=png')
    
    df = df.drop(columns=['Chg', '% Chg', "#", "Price", "Weight"])
    
    data = df.to_dict(orient='records')
    #file_path = 'reduced_sp500_tickers.json'
    #with open(file_path, 'w') as json_file:
    #    json.dump(data, json_file, indent=4)

    return data

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
        
@router.get('/getTickerDataFromDate', summary="Get ticker data from a certain date")
async def getTickerDataFromDate(symbol: str, date: str):
    repo = TicketRepo()
    
    ticket = await repo.searchValueOfTicketFromDate(symbol, date)
    return ticket
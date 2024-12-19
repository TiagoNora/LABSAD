from fastapi import APIRouter, Depends, HTTPException
import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup as bs
import os
import json
from labsadbackend.repo import *
import functools
from pymongo import MongoClient
import matplotlib.pyplot as plt
import uuid


router = APIRouter(prefix='/tickers', tags=['TICKERS'])

@router.get('/relatedStocks', summary="Get related stocks")
async def getRelatedStocks(symbol: str):
    ticker = yf.Ticker(symbol)
    sector = yf.Sector(ticker.info.get('sectorKey'))
    name_dict = sector.top_companies.get("name", {})
    transformed = [{"symbol": key, "name": value} for key, value in name_dict.items()]
    return {"tickers": transformed[:5]}


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

@router.get('/historialData1Week', summary="Get ticker data 1 week")
async def getTicker1Week(symbol: str):
    ticker = yf.Ticker(symbol)
    tickerPrices = ticker.history(period="5d")
    tickerPrices.reset_index(inplace=True)
    return tickerPrices.to_dict(orient='records')

@router.get('/historialDataYtd', summary="Get ticker data year to date")
async def getTicker1Week(symbol: str):
    ticker = yf.Ticker(symbol)
    tickerPrices = ticker.history(period="ytd")
    tickerPrices.reset_index(inplace=True)
    return tickerPrices.to_dict(orient='records')

@router.get('/historialData1Month', summary="Get ticker data 1 month")
async def getTicker1Month(symbol: str):
    ticker = yf.Ticker(symbol)
    tickerPrices = ticker.history(period="1mo")
    tickerPrices.reset_index(inplace=True)
    return tickerPrices.to_dict(orient='records')

@router.get('/historialData1Year', summary="Get ticker data 1 year")
async def getTicker1Year(symbol: str):
    ticker = yf.Ticker(symbol)
    tickerPrices = ticker.history(period="1y")
    tickerPrices.reset_index(inplace=True)
    return tickerPrices.to_dict(orient='records')

@functools.cache
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
    
    df = df.drop(columns=['Chg', '% Chg', 'Price'])
    df = df.fillna(0)
    
    df['Image'] = df['Symbol'].apply(lambda symbol: f'https://assets.parqet.com/logos/symbol/{symbol}?format=png')
    
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

@router.get('/getInfo', summary="Get ticker info")
async def getTickerInfo(symbol: str):
    ticker = yf.Ticker(symbol)
    info = ticker.info
    info["companyOfficers"] = info.get("companyOfficers", [])[:3]
    return info

@router.get('/getNews', summary="Get ticker news")
async def getTickerNews(symbol: str):
    ticker = yf.Ticker(symbol)
    news = ticker.news
    return news

@router.get('/getRecommendations', summary="Get ticker recommendations")
async def getTickerRecommendations(symbol: str):
    ticker = yf.Ticker(symbol)
    recommendations = ticker.recommendations
    recommendations = recommendations.reset_index()
    recommendations_dict = recommendations.to_dict(orient='records')
    return recommendations_dict

@router.get('/getInstitutionalHolders', summary="Get ticker institutional holders")
async def getTickerInstitutionalHolders(symbol: str):
    ticker = yf.Ticker(symbol)
    institutional_holders = ticker.institutional_holders
    institutional_holders = institutional_holders.reset_index()
    institutional_holders_dict = institutional_holders.head(5).to_dict(orient='records')
    return institutional_holders_dict

@router.get('/searchPortfolio', summary="Search for a ticker for portfolio")
async def searchTicker(name: str):
    matching_tickers = [company for company in tickers if company["Company"].lower().startswith(name.lower())]

    if not matching_tickers:
        raise HTTPException(status_code=404, detail="No companies found for the provided ticker")

    # Return a list of matching tickers and names
    return matching_tickers[:5]

@router.get('/getForecasts', summary="Obtain 7 days of forecasts for a given company")
async def getForecasts7Days(symbol: str):
    
    uri = 'mongodb://mongoadmin:a79c987b4dce244e9bc21620@vsgate-s1.dei.isep.ipp.pt:10777'  # MongoDB connection URI
    db_name = 'labsad'  # Replace with your database name
    stocks_forecast = 'stocksForecasts'

    # # Initialize MongoDB client
    client = MongoClient(uri)
    db = client[db_name]
    stocks_forecasts_collection = db[stocks_forecast]
    document = stocks_forecasts_collection.find_one({"symbol": symbol})

    if document:
        # If the document exists, return the 7 predictions
        predictions = document.get("predictions", [])
        client.close()
        return {
                "symbol": symbol,
                "predictions": predictions
            }
    else:
        print(f"No document found for symbol: {symbol}")
        return None
    


@router.get('/getForecastsImage', summary="Obtain 7 days of forecasts for a given company")
async def getForecasts7DaysAfterBefore(symbol: str):
    uri = 'mongodb://mongoadmin:a79c987b4dce244e9bc21620@vsgate-s1.dei.isep.ipp.pt:10777'
    db_name = 'labsad'
    save_path = 'assets/images'

    # Initialize MongoDB client
    client = MongoClient(uri)
    db = client[db_name]

    
    # Fetch forecasted data
    stocks_forecast = 'stocksForecasts'
    stocks_forecasts_collection = db[stocks_forecast]
    document = stocks_forecasts_collection.find_one({"symbol": symbol})

    if not document or "predictions" not in document:
        raise ValueError(f"No forecasts found for symbol: {symbol}")

    predictions = document["predictions"]

    # Fetch actuals from Yahoo Finance
    actuals = fetch_actuals_from_yahoo(symbol)

    print(symbol)
    print(actuals)
    print(predictions)
    print(save_path)

    # Plot
    file_image =  plot_forecast_with_actuals(symbol, actuals, predictions, save_path)

    url = "https://labsad.onrender.com/" + file_image  # Replace with your FastAPI server URL


    return {'Image': url}

     





##### functions




def plot_forecast_with_actuals(symbol, actuals, predictions, save_path):
    """
    Plots the last 7 actual values and forecasted values for a stock symbol.

    Parameters:
    - symbol: str, stock symbol.
    - actuals: list, last 7 actual values.
    - predictions: list, forecasted values.
    - save_path: str, path to save the plot.
    """
    if len(actuals) != 7:
        raise ValueError("The actuals list must contain exactly 7 values.")

    plt.figure(figsize=(10, 6))
    print('hello')
    # Plot actuals
    days = [f"Day-{i}" for i in range(7, 0, -1)]
    plt.plot(days, actuals, marker='o', label='Last 7 Actuals', color='blue')

    print(predictions)
    # Plot forecasts
    prediction_values = [p['prediction'] for p in predictions]
    prediction_dates = [p['date'] for p in predictions]
    plt.plot(prediction_dates, prediction_values, marker='o', label='Forecasts', color='orange')

    print('ola')

    # Adding labels and title
    plt.xlabel('Days')
    plt.ylabel('Stock Value')
    plt.title(f'{symbol} Actuals vs Forecasts')
    plt.grid(True)
    plt.legend()

    # Save plot
    plt.tight_layout()

    # Ensure the output directory exists
    os.makedirs(save_path, exist_ok=True)
    unique_filename = f"{uuid.uuid4()}" + ".png"
    
    # Define the file path for saving the image (e.g., PNG format)
    file_path = os.path.join(save_path, unique_filename)


    plt.savefig(save_path)
    #plt.close()

    return file_path

def fetch_actuals_from_yahoo(symbol):
    """
    Fetches the last 7 actual values for a stock symbol from Yahoo Finance.

    Parameters:
    - symbol: str, stock symbol.

    Returns:
    - list, last 7 actual values.
    """
    try:
        # Create a Ticker object
        ticker = yf.Ticker(symbol)

        # Fetch historical data for the last month
        data = ticker.history(period="1mo", interval="1d")

        if data.empty:
            raise ValueError(f"No data found for symbol: {symbol}")

        # Get the 'Close' prices for the most recent 7 days
        actuals = data['Close'].tail(7).tolist()
        return actuals
    except Exception as e:
        raise ValueError(f"Error fetching data from Yahoo Finance for symbol {symbol}: {e}")



    
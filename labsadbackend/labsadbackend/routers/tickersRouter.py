from fastapi import APIRouter, Depends, HTTPException
import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup as bs
import os
import json
from fastapi.responses import JSONResponse
from datetime import datetime, timedelta
from labsadbackend.repo import *
import functools
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING messages
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping

import joblib


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
    institutional_holders_dict = institutional_holders.to_dict(orient='records')
    return institutional_holders_dict


@router.get('/getStockForecast', summary='Get 7 days of future predictions for a specific stock')
async def getForecastsTicker(symbol: str):
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Build the absolute path to the model
    model_path = os.path.join(script_dir, "models/lstm_stock_model.h5")
    scalerX_path = os.path.join(script_dir, "models/scaler_X.pkl")
    scalerY_path = os.path.join(script_dir, "models/scaler_Y.pkl")
    # Load the model
    model = load_model(model_path)
    scaler_X = joblib.load(scalerX_path)
    scaler_Y = joblib.load(scalerY_path)
    ticker = yf.Ticker(symbol)
    df = ticker.history(period="max")

    df.drop(columns=['Dividends', 'Stock Splits'], inplace=True) #these columns won't be needed for forecasts

    # Ensure the 'Date' column is of string type
    df.reset_index(inplace=True)
    df = df.dropna()
    df['Date'] = df['Date'].astype(str)

    # Now, slice the string to get just the 'YYYY-MM-DD' part
    df['Date'] = df['Date'].str[:10]
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True )
    df.head()

    X, Y = df_to_X_Y(df)

    X_full_scaled = scaler_X.fit_transform(X.reshape(-1, X.shape[2])).reshape(X.shape)
    Y_full_scaled = scaler_Y.fit_transform(Y.reshape(-1, 1))

    early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    history = model.fit(X_full_scaled, Y_full_scaled, epochs=40, batch_size=32, callbacks=[early_stopping], verbose=1)

    # Predict 7 days into the future
    last_window = X_full_scaled[-1]  # Last window for starting the prediction
    future_predictions_scaled = []

    for _ in range(7):  # Predict for 7 days
        last_window_batch = np.expand_dims(last_window, axis=0)  # Add batch dimension
        next_prediction_scaled = model.predict(last_window_batch)[0][0]
        future_predictions_scaled.append(next_prediction_scaled)
        next_row = last_window[-1].copy()  # Copy the last row
        next_row[-1] = next_prediction_scaled  # Update with the predicted value
        last_window = np.vstack([last_window[1:], next_row])  # Slide the window

    # Inverse transform the predictions
    future_predictions = scaler_Y.inverse_transform(np.array(future_predictions_scaled).reshape(-1, 1))

    # Combine historical data and future predictions
    historical_data = df['Close'].values  # Original 'Close' column
    future_time_steps = range(len(historical_data), len(historical_data) + 7)  # Future time indices

    #print(future_predictions)
    future_predictions_list = future_predictions.tolist()
    #print(future_predictions_list)
    return JSONResponse(content={"future_predictions": future_predictions_list})




@router.get('/searchPortfolio', summary="Search for a ticker for portfolio")
async def searchTicker(name: str):
    matching_tickers = [company for company in tickers if company["Company"].lower().startswith(name.lower())]

    if not matching_tickers:
        raise HTTPException(status_code=404, detail="No companies found for the provided ticker")

    # Return a list of matching tickers and names
    return matching_tickers[:5]



def df_to_X_Y(df, window_size=8):
    df_as_np = df.to_numpy()
    X = []
    Y = []
    for i in range(len(df_as_np)-window_size):
        row = [r for r in df_as_np[i:i+window_size]]
        X.append(row)
        label = df_as_np[i+window_size][3]
        Y.append(label)
    return np.array(X), np.array(Y)
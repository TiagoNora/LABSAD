import uvicorn
from fastapi import FastAPI
from fastapi_utilities import repeat_at, repeat_every
import logging
from fastapi import APIRouter, Depends, HTTPException
import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup as bs
import os
import json
from fastapi.responses import JSONResponse
from datetime import datetime, timedelta
import functools
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING messages
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
import joblib


app = FastAPI(title="api")

logger = logging.getLogger('uvicorn.error')

@app.on_event('startup')
#@repeat_every(seconds=1)
async def print_hello():
    #logger.debug('this is a debug message')

    symbol = 'A'
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
    logger.debug(future_predictions_list)



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









@app.get("/")
async def read_root():
    return {"Hello": "World"}

def start():
    uvicorn.run("labsadmodels.main:app", host="0.0.0.0", port=8000, workers=4, reload=True, log_level="debug")



from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect, File, UploadFile
import asyncio
from labsadbackend.repo import *
from labsadbackend.models import *
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import yfinance as yf
from datetime import datetime, timedelta
from fredapi import Fred
from fastapi.staticfiles import StaticFiles
import base64
import uuid
import os
import matplotlib.pyplot as plt
import requests
import random


router = APIRouter(prefix='/portfolio', tags=['PORTFOLIO'])

@router.get('/all', summary="Get all portfolios")
async def getPortfolios(email: str):
    repo = PortfolioRepo()
    portfolios = repo.getAllPortfolios(email)
    return portfolios

@router.get('/search', summary="Search for a portfolio")
async def searchPortfolio(name: str, email: str):
    repo = PortfolioRepo()
    portfolio = repo.getPortfolio(name, email)
    return portfolio

@router.post('/create', summary="Create a portfolio")
async def createPortfolio(portfolioCreate: PortfolioCreate):
    repo = PortfolioRepo()
    repo.createPortfolio(portfolioCreate)
    return {"message": "Portfolio created successfully"}

@router.put('/update', summary="Update a portfolio")
async def updatePortfolio(portfolioUpdate: PortfolioUpdate):
    repo = PortfolioRepo()
    repo.updatePortfolio(portfolioUpdate)
    return {"message": "Portfolio updated successfully"}

@router.delete('/delete', summary="Delete a portfolio")
async def deletePortfolio(portfolioDelete: PortfolioDelete):
    repo = PortfolioRepo()
    repo.deletePortfolio(portfolioDelete)
    return {"message": "Portfolio deleted successfully"}

@router.post('/addTicket', summary="Add a ticket to a portfolio")
async def addTicketToPortfolio(portfolioAddTicket: PortfolioAddTicket):
    repo = PortfolioRepo()
    repo.addTicketPortfolio(portfolioAddTicket)
    return {"message": "Ticket added successfully"}

@router.delete('/deleteTicket', summary="Delete a ticket from a portfolio")
async def deleteTicketFromPortfolio(portfolioRemoveTicket: PortfolioRemoveTicket):
    repo = PortfolioRepo()
    repo.deleteTicketPortfolio(portfolioRemoveTicket)
    return {"message": "Ticket removed successfully"}

@router.get('/portfolioOptimization', summary="Given a list a tickers, an optimization suggestion is provided")
async def optimizePortfolio(name: str, email: str):
    repo = PortfolioRepo()
    portfolio = repo.getPortfolio(name, email)
    tickerList = []
    for t in portfolio['stocks']:
        tickerList.append(t['symbol'])
    # print(tickerList)
    return optimize_stock_list(tickerList)


@router.get('/portfolioOptimizationMaxReturns', summary="Given a list a tickers, an optimization suggestion is provided with the intention of maximizing returns")
async def optimizePortfolioMaxReturns(name: str, email: str):
    repo = PortfolioRepo()
    portfolio = repo.getPortfolio(name, email)
    tickerList = []
    for t in portfolio['stocks']:
        tickerList.append(t['symbol'])
    # print(tickerList)
    return optimize_max_returns(tickerList)
    

@router.get('/portfolioOptimizationMinRisk', summary="Given a list a tickers, an optimization suggestion is provided with the intention of minimizing risks")
async def optimizePortfolioMinRisk(name: str, email: str):
    repo = PortfolioRepo()
    portfolio = repo.getPortfolio(name, email)
    tickerList = []
    for t in portfolio['stocks']:
        tickerList.append(t['symbol'])
    # print(tickerList)
    return optimize_min_risk(tickerList)
    
@router.get('/portfolioInfo', summary='Given a portfolio, return stats and info on it')
async def portfolioInfo(name: str, email:str):
    repo = PortfolioRepo()
    portfolio = repo.getPortfolio(name, email)
    portfolio = portfolio['stocks']

    sectors = fetch_sector_classification(portfolio)
    #print("Sector Classification:", sectors)

    current_prices = fetch_current_prices(portfolio)
    #print("Current Prices:", current_prices)

    total_value, sector_allocations = calculate_portfolio_stats(portfolio, current_prices, sectors)

    portfolio_return = calculate_returns(portfolio, current_prices)

    #sector_string = sector_allocations_to_string(sector_allocations)
    
    transformed = [{"sector": key, "percentage": round(value,2)} for key, value in sector_allocations.items()]


    # # Display results
    # print(f"Portfolio Total Value: ${total_value:.2f}")
    # print(f"Sector Allocations: {sector_allocations}")
    # print(f"Portfolio Return: {portfolio_return:.2f}%")

    # Convert dictionaries to lists of tuples
    current_prices = dict_to_list_of_dicts(current_prices, 'current_price')
    #sectors = dict_to_list_of_dicts(sectors, 'sector')
    
    print(sectors)
    
    transformedCategory = [{"ticker": key, "category": value} for key, value in sectors.items()]
    
    merged_data = [
    {
        "symbol": price["symbol"],
        "sector": sectors.get(price["symbol"]),
        "current_price": price["current_price"]
    }
    for price in current_prices
    ]
        

    result = {'returns':portfolio_return, 'sector_allocation': transformed, 'total_value':total_value, 'current_prices': merged_data}

    #print(result)
    return result
    
    
@router.get('/portfolioBenchmark', summary='Given a portfolio, benchmark it agaisnt other indexes')
async def portfolioBenchmark(name: str, email:str, index: str):
    repo = PortfolioRepo()
    portfolio = repo.getPortfolio(name, email)
    portfolio = portfolio['stocks']
    start_date = min(stock['buyDate'] for stock in portfolio)
    index = "^" + index
    #print(index)
    # Calculate performances
    portfolio_perf, overall_return = calculate_portfolio_performance(portfolio)
    benchmark_perf = benchmark_performance(index, start_date, datetime.now().strftime('%Y-%m-%d'))
    
    # Print results
    print("Portfolio Performance:", portfolio_perf)
    print("Overall Portfolio Return:", overall_return)
    print(f"{index} Benchmark Return:", benchmark_perf)

    file_image = plot_comparison(portfolio, index, start_date)

    print(file_image)

    url = "https://labsad.onrender.com/" + file_image  # Replace with your FastAPI server URL
    

    return {'Image': url}


@router.get('/portfolioStockPercentage', summary='Given a portfolio, return the percentage of each stock')
async def portfolioStockPercentage(name: str, email:str):
    repo = PortfolioRepo()
    portfolio = repo.getPortfolio(name, email)
    portfolio = portfolio['stocks']
    total = 0
    for stock in portfolio:
        total += stock['quantity'] * stock['buyPrice']
    for stock in portfolio:
        stock['percentage'] = round(((stock['quantity']* stock['buyPrice'])/total) * 100,2)
    return portfolio

"""UPLOAD_DIR = "Server/static"
os.makedirs(UPLOAD_DIR, exist_ok=True) 

@router.post("/upload_image")
async def upload_image(file: UploadFile = File(...)):
    try:
        # Generate a unique file name
        file_extension = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = os.path.join(UPLOAD_DIR, unique_filename)
        
        # Save the uploaded file
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Construct the accessible URL
        file_url = f"http://labsad.onrender.com/Static/static/{unique_filename}"
        return {"file_url": file_url}
    except Exception as e:
        return {"error": str(e)}
"""


@router.get('/portfolioBacktesting', summary='Given a portfolio, benchmark it agaisnt other indexes')
async def portfolioBacktesting(stock_list_str: str, weight_list_str: str):

    stock_list = [s.strip() for s in stock_list_str.split(',')]
    weight_list = [float(s) for s in weight_list_str.split(',')]
    # print(stock_list)
    # print(weight_list)
    end_date = datetime.today()
    n_years = 5
    start_date = end_date - timedelta(days = n_years * 365)

    full_portfolio_prices, file_image = portfolio_value_evaluation(stock_list, weight_list, start_date, end_date)
    total_returns = (full_portfolio_prices['Ptf Value'][-1] / full_portfolio_prices['Ptf Value'][0])-1
    print("Total portfolio return:", f"{total_returns:.2%}")

    
    url = "https://labsad.onrender.com/" + file_image  # Replace with your FastAPI server URL
    

    return {'Image': url, 'total_returns': total_returns*100}



@router.get('/portfolioStockRecommendation', summary='Given a portfolio, recommend stocks of unrepresented sectors')
async def recommendStocks(name: str, email: str):
    repo = PortfolioRepo()
    portfolio = repo.getPortfolio(name, email)
    tickerList = []
    for t in portfolio['stocks']:
        tickerList.append(t['symbol'])



    recommendations = recommend_stocks_from_new_sectors(tickerList)

    return recommendations


@router.get('/PortfolioStockRecommendationbySector', summary="For a specific sector, recommend a stock")
async def recommendStocksBySector(name: str, email: str):
    repo = PortfolioRepo()
    portfolio = repo.getPortfolio(name, email)
    existing_tickers = {t['symbol'] for t in portfolio['stocks']}

    # Fetch stock data dynamically
    try:
        stocks = fetch_stock_data()
    except Exception as e:
        return {"error": f"Unable to fetch stocks: {str(e)}"}

    # Filter out stocks already in the user's portfolio
    available_stocks = [
        {
            "symbol": stock["symbol"],
            "name": stock.get("description", "No name available"),
            "sector": stock.get("sector", "No sector information"),
        }
        for stock in stocks
        if stock["symbol"] not in existing_tickers
    ]

    if not available_stocks:
        return {"error": "No new stocks to recommend at the moment."}

    # Randomly select a stock to recommend
    recommended_stock = random.choice(available_stocks)

    # Fetch additional details about the recommended stock
    try:
        details = fetch_stock_details(recommended_stock["symbol"])
        recommended_stock["current_price"] = details.get("c", "N/A")  # Current price
        recommended_stock["day_high"] = details.get("h", "N/A")  # Day high
        recommended_stock["day_low"] = details.get("l", "N/A")  # Day low
    except Exception as e:
        recommended_stock["details_error"] = f"Failed to fetch additional details: {str(e)}"

    return recommended_stock




######## Auxiliary functions

API_KEY = "cti2qn9r01qm6mum0010cti2qn9r01qm6mum001g"  # Replace with your API key
BASE_URL = "https://finnhub.io/api/v1/"

def fetch_stock_data():
    """Fetches a list of interesting stocks dynamically from an API."""
    url = f"{BASE_URL}stock/symbol?exchange=US&token={API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to fetch stock data. HTTP {response.status_code}: {response.text}")

def fetch_stock_details(symbol):
    """Fetches detailed information for a specific stock."""
    url = f"{BASE_URL}quote?symbol={symbol}&token={API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to fetch stock details for {symbol}. HTTP {response.status_code}: {response.text}")


def get_user_sectors(ticker_list):
    """Fetch the sectors of user's stocks."""
    sectors = set()
    for ticker in ticker_list:
        stock = yf.Ticker(ticker)
        try:
            info = stock.info
            sectors.add(info.get('sector', 'Unknown'))
        except KeyError:
            # If sector info is missing, skip that stock
            continue
    return sectors

def get_all_sectors():
    """Predefined list of sectors, we may expand it later based on available data."""
    return {
        "Technology", "Healthcare", "Financials", "Energy", 
        "Consumer Discretionary", "Consumer Staples", "Industrials",
        "Materials", "Utilities", "Real Estate", "Communication Services"
    }

def get_high_growth_stocks(sector):
    """Find high-growth stocks in the given sector by checking relevant data."""
    # Example of high-growth criteria: Fast revenue growth, earnings growth, etc.
    growth_stocks = []
    
    # This part will simulate getting top stocks in that sector
    # For now, let's manually define a few popular high-growth stocks for sectors
    
    if sector == "Technology":
        growth_stocks = ["NVDA", "AMD", "MSFT", "GOOGL", "AAPL"]
    elif sector == "Healthcare":
        growth_stocks = ["BIIB", "REGN", "VRTX", "PFE", "JNJ"]
    elif sector == "Financials":
        growth_stocks = ["JPM", "GS", "MS", "C", "WFC"]
    elif sector == "Energy":
        growth_stocks = ["XOM", "CVX", "SLB", "COP", "ENB"]
    elif sector == "Consumer Discretionary":
        growth_stocks = ["TSLA", "AMZN", "HD", "NKE", "MCD"]
    elif sector == "Consumer Staples":
        growth_stocks = ["PG", "KO", "PEP", "CL", "MO"]
    elif sector == "Industrials":
        growth_stocks = ["CAT", "BA", "UPS", "DE", "CSX"]
    elif sector == "Materials":
        growth_stocks = ["LIN", "APD", "ECL", "DD", "LQD"]
    elif sector == "Utilities":
        growth_stocks = ["NEE", "DUK", "SO", "XEL", "AWK"]
    elif sector == "Real Estate":
        growth_stocks = ["PLD", "SPG", "AMT", "EQIX", "O"]
    elif sector == "Communication Services":
        growth_stocks = ["DIS", "T", "VZ", "TMUS", "NFLX"]
    
    return growth_stocks

def recommend_stocks_from_new_sectors(user_tickers):
    """Recommend high-growth stocks from sectors the user hasn't invested in."""
    user_sectors = get_user_sectors(user_tickers)
    all_sectors = get_all_sectors()
    uninvested_sectors = all_sectors - user_sectors

    recommendations_list = []

    recommendations = {}
    print(uninvested_sectors)
    for sector in uninvested_sectors:
        # Get high-growth stocks in this sector
        recommendations['sector'] = sector
        recommendations[f'stocks'] = get_high_growth_stocks(sector)
        recommendations_list.append(recommendations)
        recommendations = {}


    return recommendations_list




def plot_comparison(portfolio, index, start_date):
    """
    Visualize portfolio vs benchmark performance.
    """

    UPLOAD_DIR = "assets/images"
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)  # Ensure the directory exists

    unique_filename = f"{uuid.uuid4()}" + ".png"
    output_file = os.path.join(UPLOAD_DIR, unique_filename)

    plt.figure(figsize=(10, 6))

    # Portfolio performance over time
    for stock in portfolio:
        prices = fetch_data(stock['symbol'], stock['buyDate'], datetime.now().strftime('%Y-%m-%d'))
        plt.plot(prices.index, prices / prices.iloc[0] * 100, label=f"{stock['symbol']} (Portfolio)")

    # Benchmark performance over time
    index_prices = fetch_data(index, start_date, datetime.now().strftime('%Y-%m-%d'))
    plt.plot(index_prices.index, index_prices / index_prices.iloc[0] * 100, label=f"{index} (Benchmark)", linestyle='--')

    plt.title("Portfolio vs Benchmark Performance")
    plt.xlabel("Date")
    plt.ylabel("Normalized Price (%)")
    plt.legend()
    plt.grid()

    unique_filename = f"{uuid.uuid4()}" + ".png"
    
    # Define the file path for saving the image (e.g., PNG format)
    

    # Save the plot to a file
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")

   # plt.show()

    return output_file


def portfolio_value_evaluation(stock_names, weights, start_date, end_date):
    UPLOAD_DIR = "assets/images"
    if np.sum(weights) != 1:
        print("Sum")
    
    stock_data = yf.download(
        tickers = stock_names, 
        start = start_date,
        end = end_date)
    
    #print(weights)

    stock_prices = stock_data['Adj Close']

    weighted_stock_prices = stock_prices * weights
    stock_prices.loc[:, 'Ptf Value'] = weighted_stock_prices.sum(1)

    # Create the plot using matplotlib
    plt.figure(figsize=(10, 6))

    # Plot each stock's price and the portfolio value
    for stock in stock_names:
        plt.plot(stock_prices.index, stock_prices[stock], label=stock)

    # Plot the portfolio value
    plt.plot(stock_prices.index, stock_prices['Ptf Value'], label='Portfolio Value', linewidth=2, color='black')

    # Add title and labels
    plt.title("Portfolio Value Over Time")
    plt.xlabel("Date")
    plt.ylabel("Price (Normalized)")

    output_dir = 'assets/images'

    # Add a legend
    plt.legend()

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    unique_filename = f"{uuid.uuid4()}" + ".png"
    
    # Define the file path for saving the image (e.g., PNG format)
    file_path = os.path.join(UPLOAD_DIR, unique_filename)

    # Save the figure as a PNG image
    plt.savefig(file_path)

    # Close the plot to avoid display (if you're in a script and don't want to show it)
    plt.close()

    return stock_prices, file_path


def get_plot_data_new(portfolio, index, start_date):
 
    data_return = []  # Initialize an empty list to store the data
    
    # Portfolio performance over time
    for stock in portfolio:
        symbol = stock['symbol']
        buy_date = stock['buyDate']
        
        # Fetch stock prices
        prices = fetch_data(symbol, buy_date, datetime.now().strftime('%Y-%m-%d'))
        
        # Normalize prices to start at 100
        normalized_prices = prices / prices.iloc[0] * 100
        
        # Flatten y values if needed
        y_flattened = normalized_prices.values.flatten().tolist()
        
        # Add the stock data to the data_return list
        data_return.append({
            "symbol": symbol,
            "x": normalized_prices.index.tolist(),  # Dates as a list
            "y": y_flattened  # Flattened list of performance values
        })

    # Benchmark performance over time
    index_prices = fetch_data(index, start_date, datetime.now().strftime('%Y-%m-%d'))
    
    # Normalize index prices to start at 100
    index_returns = index_prices / index_prices.iloc[0] * 100
    
    # Flatten y values if needed
    y_flattened_index = index_returns.values.flatten().tolist()
    
    # Add the benchmark data to the data_return list
    data_return.append({
        "symbol": index,
        "x": index_returns.index.tolist(),  # Dates as a list
        "y": y_flattened_index  # Flattened list of performance values
    })

    return data_return




def fetch_data(symbol, start_date, end_date):
    """
    Fetch historical price data for a given symbol from Yahoo Finance.
    """
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    return stock_data['Adj Close']

def calculate_portfolio_performance(portfolio):
    """
    Calculate the portfolio performance based on current prices.
    """
    portfolio_performance = []
    total_investment = 0
    portfolio_value = 0

    for stock in portfolio:
        symbol = stock['symbol']
        quantity = stock['quantity']
        buy_price = stock['buyPrice']
        buy_date = stock['buyDate']
        buy_date = datetime.strptime(buy_date, '%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')

        # Fetch historical data
        prices = fetch_data(symbol, buy_date, end_date)
        current_price = prices.iloc[-1]

        # Calculate performance
        investment = buy_price * quantity
        current_value = current_price * quantity
        return_percentage = ((current_price - buy_price) / buy_price) * 100

        portfolio_performance.append({
            'symbol': symbol,
            'investment': investment,
            'current_value': current_value,
            'return_percentage': return_percentage
        })

        total_investment += investment
        portfolio_value += current_value

    overall_return = ((portfolio_value - total_investment) / total_investment) * 100

    return portfolio_performance, overall_return

def benchmark_performance(index, start_date, end_date):
    """
    Calculate benchmark performance over the same period.
    """
    index_data = fetch_data(index, start_date, end_date)
    start_price = index_data.iloc[0]
    end_price = index_data.iloc[-1]
    return ((end_price - start_price) / start_price) * 100



# Function to transform sector allocation dictionary to the string format
def sector_allocations_to_string(sector_allocations):
    # Convert the dictionary to a list of strings in "sector:number" format
    sector_string = ','.join([f"{sector}:{allocation:.2f}" for sector, allocation in sector_allocations.items()])
    return sector_string

def dict_to_list_of_dicts(dictionary, value_key):
    return [{'symbol': key, value_key: value} for key, value in dictionary.items()]


# Function to fetch sector classifications
def fetch_sector_classification(portfolio):
    sector_classification = {}
    for stock in portfolio:
        symbol = stock['symbol']
        try:
            ticker_info = yf.Ticker(symbol).info  # Retrieve stock info
            sector = ticker_info.get('sector', 'Unknown')  # Fetch sector
            sector_classification[symbol] = sector
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            sector_classification[symbol] = 'Unknown'
    return sector_classification

# Fetch current stock data
def fetch_current_prices(portfolio):
    prices = {}
    for stock in portfolio:
        symbol = stock['symbol']
        try:
            data = yf.Ticker(symbol).history(period="1d")  # Get latest price
            current_price = data['Close'].iloc[-1]  # Close price
            prices[symbol] = current_price
        except Exception as e:
            print(f"Error fetching price for {symbol}: {e}")
            prices[symbol] = 0
    return prices

# Calculate portfolio value and sector allocation
def calculate_portfolio_stats(portfolio, prices, sectors):
    total_value = 0
    sector_values = {}

    for stock in portfolio:
        symbol = stock['symbol']
        quantity = stock['quantity']
        current_price = prices[symbol]
        stock_value = quantity * current_price

        total_value += stock_value
        sector = sectors.get(symbol, 'Unknown')
        if sector not in sector_values:
            sector_values[sector] = 0
        sector_values[sector] += stock_value

    sector_allocations = {sector: (value / total_value) * 100 for sector, value in sector_values.items()}
    return total_value, sector_allocations

def calculate_returns(portfolio, prices):
    total_cost = sum(stock['quantity'] * stock['buyPrice'] for stock in portfolio)
    current_value = sum(stock['quantity'] * prices[stock['symbol']] for stock in portfolio)
    return ((current_value - total_cost) / total_cost) * 100


def optimize_stock_list(tickerList):
    end_date = datetime.today()
    start_date = end_date - timedelta(days = 5*365) #5 years to look back and analyze
    adj_close_df = pd.DataFrame()
    for ticker in tickerList:
        data = yf.download(ticker, start = start_date,end = end_date)
        adj_close_df[ticker] = data['Adj Close']
    log_returns = np.log(adj_close_df / adj_close_df.shift(1))
    log_returns = log_returns.dropna()
    cov_matrix = log_returns.cov() * 252
    fred = Fred(api_key='ec3681199eb0a233f68e813ebff4cb55')
    ten_year_treasury_rate = fred.get_series_latest_release('GS10') / 100
    risk_free_rate = ten_year_treasury_rate.iloc[-1]
    constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
    bounds = [(0, 0.4) for _ in range(len(tickerList))]
    initial_weights = np.array([1/len(tickerList)]*len(tickerList))

    optimized_results = minimize(neg_sharpe_ratio, initial_weights, args=(log_returns, cov_matrix, risk_free_rate), method='SLSQP', constraints=constraints, bounds=bounds)

    optimal_weights = optimized_results.x

    print("Optimal Weights:")
    for ticker, weight in zip(tickerList, optimal_weights):
        print(f"{ticker}: {weight:.4f}")

    optimal_portfolio_return = expected_return(optimal_weights, log_returns)
    optimal_portfolio_volatility = standard_deviation(optimal_weights, cov_matrix)
    optimal_sharpe_ratio = sharpe_ratio(optimal_weights, log_returns, cov_matrix, risk_free_rate)

   
     # Create a dictionary to hold the data
    result = {
        "Optimal Weights": [],
        "Optimal Portfolio Return": round(optimal_portfolio_return, 4),
        "Optimal Portfolio Volatility": round(optimal_portfolio_volatility, 4),
        "Optimal Sharpe Ratio": round(optimal_sharpe_ratio, 4)
    }


    for n in range(len(tickerList)):
        ticker_weight = {'name':tickerList[n], 'weight':round(optimal_weights[n], 2)}
        result['Optimal Weights'].append(ticker_weight)

    return result


def standard_deviation(weights, cov_matrix):
    variance = weights.T @ cov_matrix @ weights
    return np.sqrt(variance)

def expected_return(weights, log_returns):
    return np.sum(log_returns.mean()*weights)*252

def sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate):
    return (expected_return(weights, log_returns) - risk_free_rate) / standard_deviation(weights, cov_matrix)

def neg_sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate):
    return -sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate)




# Function for maximizing returns
def optimize_max_returns(tickerList):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=5*365)  # 5 years look-back period
    adj_close_df = pd.DataFrame()
    for ticker in tickerList:
        data = yf.download(ticker, start=start_date, end=end_date)
        adj_close_df[ticker] = data['Adj Close']
    log_returns = np.log(adj_close_df / adj_close_df.shift(1))
    log_returns = log_returns.dropna()
    
    # Constraints: weights must sum to 1
    constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
    bounds = [(0, 0.4) for _ in range(len(tickerList))]
    initial_weights = np.array([1 / len(tickerList)] * len(tickerList))

    
    # Negating expected return to maximize
    def neg_expected_return(weights):
        return -expected_return(weights, log_returns)
        
    
    
    optimized_results = minimize(neg_expected_return, initial_weights, method='SLSQP', constraints=constraints, bounds=bounds)
    optimal_weights = optimized_results.x
    
    # Portfolio metrics
    optimal_portfolio_return = expected_return(optimal_weights, log_returns)
    optimal_portfolio_volatility = standard_deviation(optimal_weights, log_returns.cov() * 252)
 
    
    result = {
        "Optimal Weights": [],
        "Optimal Portfolio Return": round(optimal_portfolio_return, 4),
        "Optimal Portfolio Volatility": round(optimal_portfolio_volatility, 4),
    }

    for n in range(len(tickerList)):
        ticker_weight = {'name':tickerList[n], 'weight':round(optimal_weights[n], 2)}
        result['Optimal Weights'].append(ticker_weight)

    return result




# Function for minimizing risk
def optimize_min_risk(tickerList):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=5*365)  # 5 years look-back period
    adj_close_df = pd.DataFrame()
    for ticker in tickerList:
        data = yf.download(ticker, start=start_date, end=end_date)
        adj_close_df[ticker] = data['Adj Close']
    log_returns = np.log(adj_close_df / adj_close_df.shift(1))
    log_returns = log_returns.dropna()
    cov_matrix = log_returns.cov() * 252
    
    # Constraints: weights must sum to 1
    constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
    bounds = [(0, 0.4) for _ in range(len(tickerList))]
    initial_weights = np.array([1 / len(tickerList)] * len(tickerList))
    
    # Minimize standard deviation
    def portfolio_volatility(weights):
        return standard_deviation(weights, cov_matrix)
    
    optimized_results = minimize(portfolio_volatility, initial_weights, method='SLSQP', constraints=constraints, bounds=bounds)
    optimal_weights = optimized_results.x
    
    # Portfolio metrics
    optimal_portfolio_return = expected_return(optimal_weights, log_returns)
    optimal_portfolio_volatility = standard_deviation(optimal_weights, cov_matrix)


    result = {
        "Optimal Weights": [],
        "Optimal Portfolio Return": round(optimal_portfolio_return, 4),
        "Optimal Portfolio Volatility": round(optimal_portfolio_volatility, 4),
    }

    for n in range(len(tickerList)):
        ticker_weight = {'name':tickerList[n], 'weight':round(optimal_weights[n], 2)}
        result['Optimal Weights'].append(ticker_weight)

    return result




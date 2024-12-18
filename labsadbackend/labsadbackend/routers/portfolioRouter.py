from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
import asyncio
from labsadbackend.repo import *
from labsadbackend.models import *
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import yfinance as yf
from datetime import datetime, timedelta
from fredapi import Fred


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
async def portfolioInfo(name: str, email:str, index: str):
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







    return get_plot_data_new(portfolio, index, start_date)








######## Auxiliary functions


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




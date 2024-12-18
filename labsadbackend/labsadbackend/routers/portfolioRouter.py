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
    
@router.get('portfolioInfo', summary='Given a portfolio, return stats and info on it')
async def portfolioInfo(name: str, email:str):
    repo = PortfolioRepo()
    portfolio = repo.getPortfolio(name, email)
    portfolio = portfolio['stocks']
    # Fetch sector classifications
    sectors = fetch_sector_classification(portfolio)
    print("Sector Classification:", sectors)

    # Fetch current prices
    current_prices = fetch_current_prices(portfolio)
    print("Current Prices:", current_prices)

    # Calculate portfolio stats
    total_value, sector_allocations = calculate_portfolio_stats(portfolio, current_prices, sectors)

    # Calculate portfolio returns
    portfolio_return = calculate_returns(portfolio, current_prices)

    sector_string = sector_allocations_to_string(sector_allocations)


    # # Display results
    # print(f"Portfolio Total Value: ${total_value:.2f}")
    # print(f"Sector Allocations: {sector_allocations}")
    # print(f"Portfolio Return: {portfolio_return:.2f}%")

    # Convert dictionaries to lists of tuples
    current_prices = dict_to_list_of_dicts(current_prices, 'current_price')
    sectors = dict_to_list_of_dicts(sectors, 'sector')

    result = {'returns':portfolio_return, 'sector_allocation':sector_string, 'total_value':total_value, 'current_prices':current_prices, 'sectors':sectors   }

    print(result)
    return result
    
    















######## Auxiliary functions



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

        # Update total and sector values
        total_value += stock_value
        sector = sectors.get(symbol, 'Unknown')
        if sector not in sector_values:
            sector_values[sector] = 0
        sector_values[sector] += stock_value

    # Calculate sector allocations
    sector_allocations = {sector: (value / total_value) * 100 for sector, value in sector_values.items()}
    return total_value, sector_allocations

# Calculate portfolio returns
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




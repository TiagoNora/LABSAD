import pymongo
from datetime import datetime
import yfinance as yf
from datetime import datetime, timedelta
from pymongo import MongoClient, InsertOne
from fastapi.encoders import jsonable_encoder
from labsadbackend.models import *

uri = 'mongodb://mongoadmin:a79c987b4dce244e9bc21620@vsgate-s1.dei.isep.ipp.pt:10777'
db_name = 'portfolio' 
collection_name = 'portfolio' 
client = MongoClient(uri)
db = client[db_name]
collection = db[collection_name]

class PortfolioRepo:
    def __init__(self):
        pass
    def createPortfolio(self, portfolioCreate: PortfolioCreate):
        portfolio = Portfolio(name=portfolioCreate.name, description=portfolioCreate.description, email=portfolioCreate.email, stocks=[])
        collection.insert_one(portfolio.dict())
        
    def getAllPortfolios(self, email: str):
        portfolios = collection.find({"email": email})
        portfolios_list = []
        for portfolio in portfolios:
            portfolio["_id"] = str(portfolio["_id"])  # Converte o ObjectId para string
            portfolios_list.append(portfolio)
        return jsonable_encoder(portfolios_list)

    def getPortfolio(self, name: str, email: str):
        portfolio = collection.find_one({"name": name, "email": email})
        portfolio["_id"] = str(portfolio["_id"])
        return jsonable_encoder(portfolio)
    
    def updatePortfolio(self, portfolioUpdate: PortfolioUpdate):
        if portfolioUpdate.newName is not None and portfolioUpdate.newDescription is not None:
            collection.update_one({"name": portfolioUpdate.name, "email": portfolioUpdate.email}, 
                                  {"$set": {"name": portfolioUpdate.newName, "description": portfolioUpdate.newDescription}})
            
        elif portfolioUpdate.newName is not None and portfolioUpdate.newDescription is  None:
            collection.update_one({"name": portfolioUpdate.name, "email": portfolioUpdate.email}, 
                                  {"$set": {"name": portfolioUpdate.newName}})
            
        elif portfolioUpdate.newName is None and portfolioUpdate.newDescription is not None:
            collection.update_one({"name": portfolioUpdate.name, "email": portfolioUpdate.email}, 
                                  {"$set": {"description": portfolioUpdate.newDescription}})
            
        elif portfolioUpdate.newName is None and portfolioUpdate.newDescription is None:
            return {"message": "No changes were made"}
    
    def deletePortfolio(self, portfolioDelete: PortfolioDelete):
        collection.delete_one({"name": portfolioDelete.name, "email": portfolioDelete.email})
    
    def addTicketPortfolio(self, portfolioAddTicket: PortfolioAddTicket):
        collection.update_one({"name": portfolioAddTicket.name, "email": portfolioAddTicket.email},
                              {"$push": {"stocks": portfolioAddTicket.stock.dict()}})
    
    def deleteTicketPortfolio(self, portfolioRemoveTicket: PortfolioRemoveTicket):
        collection.update_one({"name": portfolioRemoveTicket.name, "email": portfolioRemoveTicket.email},
                              {"$pull": {"stocks": portfolioRemoveTicket.stock.dict()}})
from fastapi import APIRouter, Depends, HTTPException
from labsadbackend.repo import *
from labsadbackend.models import *

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

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
async def searchPortfolio(name: str):
    pass

@router.post('/create', summary="Create a portfolio")
async def createPortfolio(portfolioCreate: PortfolioCreate):
    repo = PortfolioRepo()
    repo.createPortfolio(portfolioCreate)
    return {"message": "Portfolio created successfully"}

@router.put('/update', summary="Update a portfolio")
async def updatePortfolio():
    pass

@router.delete('/delete', summary="Delete a portfolio")
async def deletePortfolio():
    pass

@router.post('/addTicket', summary="Add a ticket to a portfolio")
async def addTicketToPortfolio():
    pass

@router.delete('/deleteTicket', summary="Delete a ticket from a portfolio")
async def deleteTicketFromPortfolio():
    pass

@router.get('/historialData', summary="Get portfolio data")
async def getPortfolioData(name: str):
    pass
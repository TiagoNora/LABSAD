from pydantic import BaseModel
from typing import List, Optional

class PortfolioCreate(BaseModel):
    name: str
    description: str
    email: str

class Stock(BaseModel):
    symbol: str
    quantity: int
    buyPrice: float
    buyDate: str

class Portfolio(BaseModel):
    name: str
    description: str
    email: str
    stocks: List[Stock]

class PortfolioUpdate(BaseModel):
    name: str
    newName: Optional[str]
    email: str
    newDescription: Optional[str]
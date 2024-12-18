import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import tickersRouter
from .routers import portfolioRouter
from fastapi.middleware.gzip import GZipMiddleware
from fastapi import FastAPI
from fastapi.responses import UJSONResponse
from fastapi.staticfiles import StaticFiles
import os


app = FastAPI(default_response_class=UJSONResponse)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

app.include_router(tickersRouter.router)
app.include_router(portfolioRouter.router)
IMAGEDIR = os.getcwd()

app.mount("/Static", StaticFiles(directory="Server"), name="Static")

def start():
    uvicorn.run("labsadbackend.main:app", host="0.0.0.0", port=8000, workers=4, reload=True, log_level="info")


    
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import tickersRouter


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(tickersRouter.router)

def start():
    uvicorn.run("labsadbackend.main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")


    
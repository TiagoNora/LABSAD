import uvicorn
from fastapi import FastAPI
from fastapi_utilities import repeat_at, repeat_every
import logging

app = FastAPI(title="api")

logger = logging.getLogger('uvicorn.error')

@app.on_event('startup')
@repeat_every(seconds=1)
async def print_hello():
    logger.debug('this is a debug message')

@app.get("/")
async def read_root():
    return {"Hello": "World"}

def start():
    uvicorn.run("labsadmodels.main:app", host="0.0.0.0", port=8000, workers=4, reload=True, log_level="debug")

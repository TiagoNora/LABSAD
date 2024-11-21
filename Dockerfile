FROM python:3.10.11-slim

RUN apt-get update && apt-get install -y

RUN pip install poetry

WORKDIR /app

COPY ./labsadbackend /app

RUN poetry install

EXPOSE 8000

CMD ["poetry", "run", "start"]

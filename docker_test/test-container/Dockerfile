FROM python:latest

WORKDIR /app/src

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt && apt-get update

COPY . .


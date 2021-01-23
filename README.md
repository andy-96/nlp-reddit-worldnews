# nlp-reddit-worldnews

## Prerequisite

- Python 3.8

## Quickstart

1. Install pipenv: `pip3 install pipenv`
2. Setup virtual environment using pipenv: `pipenv --python 3.8`
3. Install all dependencies: `pipenv install`
4. Start server: `pipenv run uvicorn api.main:app`

## Run Docker

1. Build Dockerfile: `docker build --file Dockerfile --tag nlp-reddit-worldnews .`
2. Run Dockerfile: `nlp-reddit-worldnews % docker run -p 8000:8000 nlp-reddit-worldnews`

## Documentation

### Data aqcuisition

### Model
Using a transformer
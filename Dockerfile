FROM ubuntu:19.10

COPY ./api /api/api
COPY Pipfile /Pipfile
COPY Pipfile.lock /Pipfile.lock

RUN apt-get update \
    && apt-get install python3-dev python3-pip -y \
    && pip3 install pipenv \
    && pipenv install --system --deploy --ignore-pipfile \
    && python3 api/api/ml/model.py

ENV PYTHONPATH=/api
WORKDIR /api

EXPOSE 8000

ENTRYPOINT ["uvicorn"]
CMD ["api.main:app", "--host", "0.0.0.0"]
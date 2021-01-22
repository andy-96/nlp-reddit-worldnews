FROM ubuntu:18.04

COPY ./api /api/api
COPY Pipfile /Pipfile
COPY Pipfile.lock /Pipfile.lock

RUN export LC_ALL=C.UTF-8 \
    && export LANG=C.UTF-8 \
    && apt-get update \
    && apt-get install python3.8 python3-pip -y \
    && pip3 install pipenv \
    && pipenv --python 3.8 \
    && pipenv install --system --deploy --ignore-pipfile

ENV PYTHONPATH=/api
WORKDIR /api

EXPOSE 8000

ENTRYPOINT ["uvicorn"]
CMD ["api.main:app", "--host", "0.0.0.0"]
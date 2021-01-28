FROM ubuntu:20.04

COPY . /api

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN apt-get clean
RUN apt-get update
RUN apt-get install python3.8 python3-pip -y
RUN pip3 install pipenv
RUN cd /api && pipenv install --system --deploy

ENV PYTHONPATH=/api
WORKDIR /api

EXPOSE 8000

ENTRYPOINT ["pipenv", "run", "uvicorn", "api.main:app", "--host", "0.0.0.0"]
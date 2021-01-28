FROM ubuntu:20.04

COPY . /api

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
# Pass your environment variables
# ENV TF_PORT=
# ENV TF_HOST=
ENV HOST=8000

RUN apt-get update
RUN apt-get install python3.8 python3-pip -y
RUN pip3 install pipenv
RUN cd /api && pipenv install --system --deploy

ENV PYTHONPATH=/api
WORKDIR /api

EXPOSE 8000

ENTRYPOINT ["pipenv", "run", "python3", "-m", "api.main"]
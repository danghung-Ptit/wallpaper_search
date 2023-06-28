FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update
RUN apt-get install -y python3.9 python3.9-dev python3.9-venv python3-pip

ADD app/ /app
COPY requirements.txt /requirements.txt
COPY runserver.sh /runserver.sh

RUN python3.9 -m pip install --upgrade pip
RUN python3.9 -m pip install -r requirements.txt && python3.9 -m pip install gunicorn

ENTRYPOINT ["/runserver.sh"]

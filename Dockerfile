FROM python:3.9

WORKDIR /app

COPY requirements.txt .
COPY runserver.sh /app/runserver.sh

RUN pip install --upgrade pip
RUN pip install -r requirements.txt && pip install uvicorn

COPY . .

CMD ["sh", "runserver.sh"]

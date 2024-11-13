FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN apt-get update && apt-get install -y build-essential libhdf5-dev libblas-dev
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["gunicorn", "app:app"]

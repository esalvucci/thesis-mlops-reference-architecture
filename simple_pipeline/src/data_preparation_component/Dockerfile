FROM python:3

WORKDIR /app

COPY data_preparation.py .
COPY requirements.txt .

RUN pip3 install -r requirements.txt

ENTRYPOINT ["python3", "data_preparation.py"]

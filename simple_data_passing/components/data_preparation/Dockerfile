FROM python:3

WORKDIR /app

RUN mkdir src
COPY src/data_preparation.py ./src
COPY requirements.txt .

RUN pip3 install -r requirements.txt

ENTRYPOINT ["python3", "src/data_preparation.py"]

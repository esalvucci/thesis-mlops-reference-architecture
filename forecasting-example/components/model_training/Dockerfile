FROM python:3.8-slim-buster

WORKDIR /app

COPY datasets/ datasets/
COPY requirements.txt .

RUN mkdir src && pip3 install -r requirements.txt

COPY src/model_training.py src

ENTRYPOINT ["python3", "src/model_training.py"]

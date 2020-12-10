FROM python:3

WORKDIR /app

RUN mkdir src
COPY src/model_training.py ./src
COPY requirements.txt .

RUN pip3 install -r requirements.txt

ENTRYPOINT ["python3", "src/model_training.py"]

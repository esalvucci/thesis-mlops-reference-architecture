FROM python:3

WORKDIR /app

COPY model_training.py .
COPY requirements.txt .

RUN pip3 install -r requirements.txt

ENTRYPOINT ["python3", "model_training.py"]

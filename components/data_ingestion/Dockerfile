FROM python:3.8-slim-buster

WORKDIR /app/src

COPY requirements.txt ..
RUN python3 -m pip install --upgrade pip && pip3 install -r ../requirements.txt
COPY src/ ./

ENTRYPOINT ["python3", "data_ingestion.py"]

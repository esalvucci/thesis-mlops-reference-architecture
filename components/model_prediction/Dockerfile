FROM python:3.8-slim-buster

WORKDIR /app/src

COPY requirements.txt ..

RUN pip3 install -r ../requirements.txt

COPY src/*.py ./

ENTRYPOINT ["python3", "model_prediction.py"]

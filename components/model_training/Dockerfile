FROM python:3.8-slim-buster

WORKDIR /app/

COPY requirements.txt ..

RUN pip3 install -r ../requirements.txt
RUN mkdir src

COPY src/*.py src
COPY MLProject .

ENTRYPOINT ["mlflow", "run"]

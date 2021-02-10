FROM python:3.8-slim-buster

WORKDIR /app

RUN mkdir src

COPY src/ src/
COPY requirements.txt .
COPY docker_endpoint.sh .

RUN apt update -y && apt install unzip gcc -y
RUN python3 -m pip install --upgrade pip
RUN pip3 install -r requirements.txt

RUN ["chmod", "+x", "docker_endpoint.sh"]
ENTRYPOINT ["/bin/bash", "docker_endpoint.sh"]
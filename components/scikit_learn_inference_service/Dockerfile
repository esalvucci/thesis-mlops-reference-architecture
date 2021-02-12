FROM python:3.8-slim-buster

WORKDIR /app

COPY requirements.txt ..

RUN mkdir /tmp/bentoservice
RUN apt update -y && apt install zip git wget gcc -y
RUN python3 -m pip install --upgrade pip && pip3 install -r ../requirements.txt

# Download and install conda, required by mlflow
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -b
RUN rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH="/root/miniconda3/bin:${PATH}"

RUN mkdir src
RUN mkdir config

COPY config/ config/
COPY src/ src/
COPY MLProject ./
COPY docker_endpoint.sh ./

RUN ["chmod", "+x", "docker_endpoint.sh"]
ENTRYPOINT ["/bin/bash", "docker_endpoint.sh"]
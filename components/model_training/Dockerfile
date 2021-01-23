FROM python:3.8-slim-buster

WORKDIR /app/

COPY requirements.txt ../
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git wget

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -b
RUN rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH="/root/miniconda3/bin:${PATH}"

RUN pip3 install -r ../requirements.txt
RUN mkdir src

COPY src/ src/
COPY MLProject ./
COPY run_mlflow.sh ./

RUN ["chmod", "+x", "run_mlflow.sh"]
ENTRYPOINT ["/bin/bash", "run_mlflow.sh"]
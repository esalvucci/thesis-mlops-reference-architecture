#!/bin/bash

# The Kubeflow pipeline will trigger this script in this way:
# /bin/bash docker_endpoint.sh (0) --dataset_path (1) <dataset path> (2) --bento_service (3) <bento service.zip> (4)
dataset_path=$2
bentoservice=$4

cp $bentoservice /tmp/bentoservice.zip
unzip /tmp/bentoservice.zip -d /tmp/service
pip3 install /tmp/service
python3 src/main.py $dataset_path
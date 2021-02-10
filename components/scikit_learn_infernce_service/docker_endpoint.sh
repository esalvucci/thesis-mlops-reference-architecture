#!/bin/bash

model_path=$2
conda_configuration_file=$4
model_metadata=$6

rm -r /tmp/bentoservice
mkdir /tmp/bentoservice

mkdir -p /tmp/model
cp $model_path /tmp/model/model.pkl
cp $conda_configuration_file /tmp/model/conda.yaml
cp $model_metadata /tmp/model/MLmodel

mlflow run . -P model_path=/tmp/model/
cd /tmp/bentoservice/
zip -r /tmp/bentoservice.zip *
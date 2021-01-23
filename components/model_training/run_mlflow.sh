#!/bin/bash

mlflow run . -P dataset_path=$2 -P n_estimators=$4 -P version=$6
#!/bin/bash

mlflow run . -P training_set_path=$2 -P test_set_path=$4 -P version=$6
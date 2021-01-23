#!/bin/bash

export DOCKER_CONTAINER_REGISTRY_BASE_URL=<CONTAINER_REGISTRY>
export MLFLOW_TRACKING_URI=<TRACKING_URI>
export TAG=<TAG>
export PROJECT_NAME='forecasting_example'
export DATA_INGESTION='data_ingestion'
export DATA_TRANSFORMATION='data_transformation'
export MODEL_TRAINING='model_training'
export MODEL_EVALUATION='model_evaluation'

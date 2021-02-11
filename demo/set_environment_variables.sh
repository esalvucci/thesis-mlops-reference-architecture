# Change DOCKER_CONTAINER_REGISTRY_BASE_URL, KUBEFLOW_HOST and MLFLOW_TRACKING_URI variables according to your
# Docker container registry url, kubeflow endpoint URL and MLFLow endpoint URL

# e.g. docker.io/repository_name or gcr.io/repository_name
export DOCKER_CONTAINER_REGISTRY_BASE_URL=<DOCKER_CONTAINER_REGISTRY>
export PROJECT_NAME='forecasting_example'
export DATA_INGESTION='data_ingestion'
export DATA_PREPARATION='data_preparation'
export BATCH_PREDICTION='scikit_learn_batch_prediction'
export INFERENCE_SERVICE='scikit_learn_inference_service'
export TAG='latest'
export MLFLOW_TRACKING_URI=<MLFLOW_TRACKING_URI> # e.g. http://34.91.32.10:5000
export KUBEFLOW_HOST=<KUBEFLOW_HOST>
export GOOGLE_APPLICATION_CREDENTIALS=<YOUR SERVICE ACCOUNT JSON FILE>
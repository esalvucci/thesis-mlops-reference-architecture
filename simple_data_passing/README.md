
## Set up your python environment

You can use conda

```
conda create --name mlpipeline python=3.7
conda activate mlpipeline
```

## Install [Kubefow Pipelines SDK](https://www.kubeflow.org/docs/pipelines/sdk/install-sdk/)

## Build docker image

On the command line run

```
export DOCKER_CONTAINER_REGISTRY_BASE_URL='your docker container registry base url'
```

The Docker container registry base url will be in the form "docker.io/username"

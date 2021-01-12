# Kubeflow examples
This example build a pipeline to make forecast predictions; it shows how
to store (and retreive data in the next step) through a bucket in Google Cloud Platform.
It also show how to use Kubeflow Reusable Components and build a 'non sequential' pipeline specifing depencencies between
the components of the pipeline.

In the following paragraph is explained how to compile and run the Kubeflow manually, refer to the
[Cloud Build](/doc/google_cloud_build) documentation to run the pipeline through an automatic trigger.

## Component folder structure
According to the [Best Practices for Designing Components](https://www.kubeflow.org/docs/pipelines/sdk/best-practices/) 
and [Organizing the component files](https://www.kubeflow.org/docs/pipelines/sdk/component-development/#organizing-the-component-files)
each component (in each example) is organized according to the following structure.

```
<component_name>/ 
    src/              # Component source code files
    Dockerfile       # Dockerfile to build the component container image
    build_image.sh   # Small script that runs docker build and docker push
    requirements.txt # The python requirements
```

## Set up your python environment
To run your component locally you can use conda. Start by setting up the conda environment.

```
conda create --name mlpipeline python=3.7
conda activate mlpipeline
```

## Install [Kubefow Pipelines SDK](https://www.kubeflow.org/docs/pipelines/sdk/install-sdk/)
To install Kubeflow Pipelines SDK on your local machine follow the instructions at the link
[Kubeflow Pipeline SDK install](https://www.kubeflow.org/docs/pipelines/sdk/install-sdk/)

## Build docker image
Each example (unless otherwise noted) use docker containers in each step.
To build an image from the Dockerfile (in each component) run the following instructions

Make sure you are logged in your docker container registry
```
sudo docker login
```

On the command line run

```
export DOCKER_CONTAINER_REGISTRY_BASE_URL='your docker container registry base url'
```

If you are using the docker container registry your base url will be in the form "docker.io/username"

In the component directory run

```
sudo -E bash -c 'source build_image.sh'
```

or if you don't need root permissions to run docker you could simply run the script (using 'source' instead of './')
The -E (or --preserve-env) argument, here, will preserve user environment variables while running the command with
root privileges

## Build the Kubeflow pipeline
In the example directory you will find two 'pipeline' folders. Move in one of them and follow the instructions below.

### Use the Kubeflow SDK to write a python file to define your pipeline
[Doc Link](https://www.kubeflow.org/docs/pipelines/sdk/)

### Compile your Kubeflow pipeline
```
dsl-compile --py pipeline.py --output pipeline.tar.gz
```

(change the input and output file names according to your pipeline python file)

## Upload and run the pipeline in Kubeflow Pipelines
Now you are ready to upload and run the pipeline on Kubeflow Pipeline
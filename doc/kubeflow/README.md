# [Kubeflow](https://kubeflow.org)
This example build a pipeline to make forecast predictions; it shows how
to retreive data from a bucket in Google Cloud Platform and pass the data between the components.
It also shows how to use Kubeflow Reusable Components and build a 'non sequential' pipeline specifing depencencies
between the components of the pipeline.

In the following paragraph is explained how to compile and run the Kubeflow pipeline manually, refer to the
[Cloud Build](/doc/google_cloud_build) and [Cloud Functions](/doc/google_cloud_functions) documentation to run the
pipeline through an automatic trigger.

## Component folder structure
As the [Best Practices for Designing Components](https://www.kubeflow.org/docs/pipelines/sdk/best-practices/)
and [Organizing the component files](https://www.kubeflow.org/docs/pipelines/sdk/component-development/#organizing-the-component-files)
suggest each component is organized according to the following structure.

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
This example uses a docker container in each step.
To build an image from the Dockerfile (in each component) run the following instructions

Make sure you are logged in your docker container registry
```
sudo docker login
```

On the command line run

```
export DOCKER_CONTAINER_REGISTRY_BASE_URL='your docker container registry base url'
```

If you are using Docker Hub as container registry your base url will be in the form "docker.io/username"

In the component directory run

```
sudo -E bash -c 'source build_image.sh'
```

or if you don't need root permissions to run docker you could simply run the script (using 'source' instead of './')
The -E (or --preserve-env) argument, here, will preserve user environment variables while running the command with
root privileges

## Build the Kubeflow pipeline
In this example you will find two 'pipeline' folders: a 'training' pipeline and a 'prediction' pipeline.
Move in one of the two directories and follow the instructions below.

### Use the Kubeflow SDK to write a python file to define your pipeline
You can find the pipelines code in this example in the {prediction_pipeline, training_pipeline}/main.py file. 

[Doc Link](https://www.kubeflow.org/docs/pipelines/sdk/)

### Compile your Kubeflow pipeline
```
dsl-compile --py main.py --output pipeline.tar.gz
```

(change the input and output file names according to your pipeline python file)

## Manually upload and run the pipeline in Kubeflow Pipelines
Now you are ready to upload and run the pipeline on Kubeflow Pipeline
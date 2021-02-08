# [Google Cloud Build](https://cloud.google.com/cloud-build)
Google Cloud Build is the CI/CD tool by Google;
it provides different actions on docker containers (build, push, deploy, etc..).

In this project Google Cloud build is used for the Continuous integration of the ML system.
A trigger in Google Cloud Build will build all the Kubeflow components (as docker containers)
and will upload and run the Kubeflow Pipeline.

Two different triggers are used, respectively, to build and run the [training pipeline](/training_pipeline/cloudbuild.yaml)
and the [prediction pipeline](/prediction_pipeline/cloudbuild.yaml).

Both the cloudbuild config file specify will trigger the following:
* Build each component of the Kubeflow Pipeline and push it to the Container Registry.
* Through the kfp-cli compile the pipeline and upload a new version of it to the Kubeflow Pipeline specified endpoint.
* Through the kfp-cli trigger the new pipeline verison run.

## Getting started 

### Prerequisites
* GitHub (or Bitbucket) repository
* Billing enabled on Google Cloud Platform
* Google Cloud Build API Enabled
* (optional) gcloud installed on your local machine
* Cluster with Kubeflow installed
    * Using the us-central1 zone you can run into some problems (403 Unauthorized)
    while uploading the Kubeflow pipeline from the trigger. A different zone for your cluster is suggested to be used.

## Build configuration

### cloudbuild.yaml
The Cloud Build yaml describes how the cloud build to be run and what arguments should be passed to the entry point
command (defined in the corresponding Dockerfile).

In the two cloud build configuration files are used docker standard cloud builders (gcr.io/cloud-builders/docker)
and a custom Cloud Build builder (gcr.io/$_PROJECT_ID/kfp-cli) which Docker file is in the [components](/components/utility/kfp-cli)
directory.

#### Steps
In each step the name is the URI of the corresponding cloud builder container;
args contain the arguments to be passed to the entry points. The 'dir' is the CWD in the Docker container from which
the entrypoint is executed. 

The 'env' property allow to pass the container environment variables (which values are specified in the trigger settings),
through the Substitution mechanism.

In the args property we build the image locally. We also need to push the built image to a docker container registry.
This operation is specified in the 'images' property.

## Execute a Cloud Build
You can execute the Cloud build either manually or through automated triggers.

Manually: Run ``` gcloud builds submit ``` on the cloudbuild.yaml with the proper substitutions.
Automated triggers: When the code thanges in your repository (for example when a push, pull request or new tag event occurs).

The [Substitution mechanism](#substitutions) is used to pass to the cloudbuild config file the values of variables
according to a specific run.

### Automated Cloud Build triggers
* Link your code repository to the Google Cloud Project.
    In the Github (or Bitbucket) marketplace, 'Apps', search "Google Cloud Build and "Set up a plan".
    Now give access to your repository by selecting it in the "Select repositories" dropdown menu.
    
![GitHub Marketplace page to link your repository to Google Cloud Build](/doc/images/gcbuild_github_marketplace.png)
* On your GCP go the the Cloud Build page and Set up your first Build Trigger selecting "Set Up Build Triggers".
  Here you can choose the event wich will invokes the trigger, the source repository and the branch wich will be monitored
  to run the trigger on (using a regular expression to match a specific branch).
  
  Make sure to also set the Cloud Build configuration file location (if your cloudbuidl.yaml file is in a specific directory)
  remember to write it down (<YOUR_DIRECTORY>/cloudbuild.yaml).
  
* Once the trigger is properly configured you can run it manually ("Run", in the "Triggers" page of the left menu)
or automatically while the selected event occurs.

Before running the trigger make sure your "pipeline" (the specified pipeline name) exists on your Kubeflow instance
(as the Experiment itself). [See this Closed Issue](https://github.com/esalvucci/mlops-architecture-example/issues/2#issue-786744338)
  
#### Substitutions Mechanism <a name = "substitutions"></a>
Substitutions, in Cloud Build, is a powerful mechanism: Cloud Build allows you to substitute variables before each of
the individual cloud builders are run.

In your cloudbuild.yaml file you can use variables with a dollar sign **followed by an underscore ('_')** and a variable
name.

You can specify the effective substitution variables on the trigger settings as showed on the following image.

![Substitution variables setting example](/doc/images/gcbuild_substitution_variables_example.png)

Substitution variables are used in both the cloudbuild config files. Default substitution variables (as $SHORT_SHA) are
documented [here](https://cloud.google.com/cloud-build/docs/configuring-builds/substitute-variable-values)

#### Container Registry
In this project the Google Container Registry is used to push the container images of each Kubeflow pipeline component.
You can watch the pushed images in the Container Registry page on GCP.
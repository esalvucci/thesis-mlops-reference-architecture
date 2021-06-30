# Thesis MLOps reference architecture
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
![GitHub issues](https://img.shields.io/github/issues/esalvucci/kubeflow-example)

## About
This project is intended to provide an example of MLOps architecture. It uses the code of a 
[Kaggle Notebook](https://www.kaggle.com/francoisraucent/forecasting-electricity-consumption-of-germany)
as use case example. The original code has been edited such to adapt it for the example in this project.

You can find the documentation about how each technology is used in the [doc](doc) folder 

In this project is used preferably Free Software (except for Google Cloud Build and Google Cloud Functions).

## Technologies
Use the following links to read the detailed documentation about how each technology is used in this project.

* [MLFlow](doc/mlflow) - Tracks the experiments log, the model versions and to store them in a Model Registry
* [Kubeflow](doc/kubeflow) - Orchestrates the ML workflow
* [BentoML](doc/bentoml) - Used as serving framework 
* Google Cloud Platform
    * [Google Cloud Build](doc/google_cloud_build) - Used to build a CI pipeline 
    * [Google Cloud Functions](doc/google_cloud_functions) - Used to run the Kubeflow pipeline whenever
a file is added or updated in the bucket used for the training dataset.

## Advantages of using MLOps
As MLOps can really improve your ML lifecycle not all the possible benefits are met in this project and highlighted here.

This project shows the following advantages and challenges you can cope by using MLOps. Each item of the list is
followed by the name (or the logo) of the technologies that address that challenge.
* ![Kubeflow](doc/images/kubeflow_logo_30x30.png) Approach to ML as a process instead of only a product
* ![Kubeflow](doc/images/kubeflow_logo_30x30.png) Reproduce the whole pipeline
* ![Kubeflow](doc/images/kubeflow_logo_30x30.png) ![MLFlow](doc/images/mlflow-logo_20x20.png) Reproduce the model building 
* ![Kubeflow](doc/images/kubeflow_logo_30x30.png) Automate the whole workflow
* ![Kubeflow](doc/images/kubeflow_logo_30x30.png) ![Google Cloud Functions](doc/images/gcf_logo_30x30.png)
  ![Google Cloud Build](doc/images/gcb_logo_30x30.png)
Auto retrain
* ![Kubeflow](doc/images/kubeflow_logo_30x30.png) Validate the model and the data (as steps of the pipeline)
* ![Kubeflow](doc/images/kubeflow_logo_30x30.png) ![Google Cloud Functions](doc/images/gcf_logo_30x30.png)
Data Drift
* ![MLFlow](doc/images/mlflow-logo_20x20.png) Increase collaboration between teams
* ![MLFlow](doc/images/mlflow-logo_20x20.png) Track the parameters used for the model training, the metrics and the model itself
* ![MLFlow](doc/images/mlflow-logo_20x20.png) Version your model
* CI/CD + CT
    * ![Kubeflow](doc/images/kubeflow_logo_30x30.png) ![Google Cloud Build](doc/images/gcb_logo_30x30.png)
    Continuous Integration
    * ![Kubeflow](doc/images/kubeflow_logo_30x30.png) ![Google Cloud Functions](doc/images/gcf_logo_30x30.png)
    Continuous Training

## Architecture
![Project Architecture](/doc/images/architecture.png)

## Licence
This project is licensed under the GPLv3 Licence - see the [LICENSE](LICENSE) file for details.
Any comment, feedback or suggestion will be appreciated

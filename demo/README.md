# Demo

## Getting started

### Datasets
You can find the datasets used in this repository in [](/demo/datasets)

### Prerequisites
To run a demo of the whole system first install
* MLFlow
* Kubeflow
* Bentoml

And set a Google Cloud Build trigger and a Google Cloud Function following the instructions in the [doc](/doc). 

### Set env variables
When your infrastructure is ready run [set_environment_variables.sh](set_environment_variables.sh) to set the env
variables required to compile the two Kubeflow Pipelines (and run MLFlow).

```
./set_env_variables.sh
```

## MLFlow
If you want to run MLFlow locally make sure to set (in the set_env_variables.sh) the URI of your MLFlow server.
Then move to the folder containing your mlflow code and

Start the mlflow ui
```
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

and run your mlflow code
```
mlflow run . # followed the required parameters
```

For example, in [/components/linear_regression_training](/components/linear_regression_training) you will run the
following command

```
mlflow run . -P dataset_path=/tmp/dataset.csv -P original_dataset_path=/tmp/it.csv
```

## Kubeflow pipelines
To compile one of the two pipelines go in the pipeline directory and run the dsl-compile command

(from the root of this repository)
```
cd training_pipeline # or cd prediction_pipeline
dsl-compile --py main.py --output pipeline.tar.gz
```

You can manually upload your compiled pipeline to Kubeflow Pipelines, create a new Experiment (or use an existing one)
and run the pipeline.

## Google Cloud Build
To set a Cloud Build trigger follow the instruction in [/doc/google_cloud_build](/doc/google_cloud_build).
You will note a new trigger in the "History" whenever you push to the selected branch in the trigger settings.

The whole Kubeflow pipeline will be rebuilt and run using the code you have just pushed.

## Google Cloud Functions
To set a Cloud Function follow the instruction in [/doc/google_cloud_functions](/doc/google_cloud_functions).
You will note a new trigger in the "History" whenever you push to the selected branch in the trigger settings.

The Kubeflow (training) pipeline will be compiled and run using, as input, the data that have been added in the target
 bucket.
 
To deploy a new Function first delete the existing one

```
gcloud functions delete <function name>
```

```gcloud functions delete run_pipeline``` according to this example.

then run 

 ```
gcloud functions deploy run_pipeline --runtime python37 --trigger-resource ${TRIGGER_BUCKET}
--trigger-event google.storage.object.finalize --env-vars-file .env.yaml
```

(where run_pipeline is the name of the function, in your python code, to be run )
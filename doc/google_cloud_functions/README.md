# Google Cloud Functions
Cloud Functions is a lightweight compute solution for developers to create single-purpose, stand-alone functions that respond to Cloud events without the need to manage a server or runtime environment. [Link](https://cloud.google.com/functions/docs/)

In this project a function is used to provide Continuous Training; whenever a new file is uploaded to the bucket
where the training dataset is stored the function triggers a new run of the Kubeflow (training) pipeline.
[Original Example](https://github.com/kubeflow/examples/blob/cookbook/cookbook/pipelines/notebooks/gcf_kfp_trigger.ipynb)


The code used for the function is in the main.py file in the 'training_pipeline' folder.

```
cd training_pipeline
```

Run the command above from the root directory of this repository.

## Requirements
Add the kfp package to your requirements.txt

```
echo "kfp" >> requirements.txt
```

## Bucket
Create a bucket in which your dataset files will be placed.
Note that in this example the function will watch (and consequently run the Kubeflow pipeline) for new file or the event
of a file update in the whole bucket and not only for some subdirectory.

Set the TRIGGER_BUCKET environment variable to your Google Cloud Storage bucket (do not include the ```gs://``` prefix
in the bucket name).

```
export TRIGGER_BUCKET=my-bucket-name
```

## The function
In the training_pipeline/main.py file you have a function named 'run_pipeline', which is the Cloud Function
to be triggered by Cloud Storage. This generic function logs relevant data when a file is changed, compiles
the training Kubeflow pipeline and runs it.

Be careful to change the HOST variable value with your Kubeflow address and the experiment named EXPERIMENT_NAME
actually exists. If you need to use IAP refer to the
[Original Example](https://github.com/kubeflow/examples/blob/cookbook/cookbook/pipelines/notebooks/gcf_kfp_trigger.ipynb)

## Deploy the function
To deploy your function run the following command:

```
gcloud functions deploy run_pipeline --runtime python37 --trigger-resource ${TRIGGER_BUCKET}
--trigger-event google.storage.object.finalize --env-vars-file .env.yaml
```

### Test your deployment

To test your deployed Google Cloud Function you can add a new file (or update an existing one) to the specified
TRIGGER_BUCKET. Then check in the [logs viewer panel](https://console.cloud.google.com/logs/viewer) to confirm that the GCF function was triggered and ran correctly.
You should also see a new run in your Kubeflow UI.

## Delete an existing pipeline

To delete an existing function run
```
gcloud functions delete <function name>
```

```gcloud functions delete run_pipeline``` according to this example.
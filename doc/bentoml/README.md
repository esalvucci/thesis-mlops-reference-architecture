# [BentoML](https://docs.bentoml.org/en/latest/)
BentoML is a framework for serving, managing, and deploying machine learning models.

In this example BentoML is used to provide and endpoint to make predictions by serving a scikit-learn based model,
on Kubernetes, through Kubeflow.

## Install

```
pip3 install bentoml scikit-learn
```
 
The code for this example is in [/components/scikitlearn_infernce_service](/components/scikit_learn_infernce_service).

## BentoML Service
The class ElectricityConsumptionRegressorService, in
[src/regressor_service.py](/components/scikit_learn_infernce_service/src/regressor_service.py), extends bentoml.BentoService,
the base class for building such prediction services using BentoML.

[[From the doc]](https://docs.bentoml.org/en/latest/concepts.html)
Each BentoService class can contain multiple models declared through the @bentoml.artifact API,
and multiple APIs for accessing this service. Each API definition requires a InputAdapter type,
which defines the expected input data format of this API. BentoML provides API input adapters
that covers most model serving use cases including DataframeInput, TfTensorInput, ImageInput
and JsonInput.

To save the BentoService instance, simply call the 'save' method, BentoML will:
* Save the model based on the ML training framework and artifact type used
* Automatically extract all the pip dependencies required by your BentoService class and put into a requirements.txt file
* Save all the local python code dependencies
* Put all the generated files into one file directory, which, by default, is a location managed by BentoML

## Create the BentoML Service
The name of the service in this example is ElectricityConsumptionRegressorService, and its code is in the
src/regressor_service.py file.

Move to your service directory by running, from the root of this repository,
```
cd components/scikit_learn_inference_service/
```

Install the requirements
```
pip3 install -r requirements.txt
```

Run your main.py file
```
python3 src/main.py
```

Use BentoML CLI tool to get the informations about your service
```
bento get ElectricityConsumptionRegressorService:latest
```

## Deploy your BentoService to Kubeflow
[Official Doc link](https://docs.bentoml.org/en/latest/deployment/kubeflow.html#deploying-to-kubeflow)

Find the local path of the latest version ElectricityConsumptionRegressorService saved bundle
```
saved_path=$(bentoml get ElectricityConsumptionRegressorService:latest --print-location --quiet)
```

Build and push the docker image of your BentoService.
Replace {docker_username} with your Docker Hub username (or gcr.io or other).

```
docker build -t <DOCKER_CONTAINER_REGISTRY>/electricity-consumption-regressor $saved_path
docker push <DOCKER_CONTAINER_REGISTRY>/electricity-consumption-regressor
```

### Kubernetes Deployment and Load Balancer
Now you can deploy your BentoService to Kubernetes. To do that you will create a new Load Balancer and its corresponding
Deployment.

The yaml file for this example is in
[electricity-consumption-regressor.yaml](/components/scikit_learn_infernce_service/electricity-consumption-regressor.yaml)

Replace <DOCKER_CONTAINER_REGISTRY> with your Docker Hub uri and username (or gcr.io or other).

Use Kubectl CLI to deploy your model server to the cluster.

```
kubectl apply -f electricity-consumption-regressor.yaml
```

Wait until your Deployment and your Load Balancer are available and, then, test your prediction endpoint with curl
```
curl -i --request POST --header "Content-Type: application/json" --data <your data> \
<endpoint ip>:<endpoint port>/predict
```

You alternatively can try your endpoint by the OpenAPI UI provided by BentoML.
Go with your browser at ```<endpoint ip>:<endpoint port>```

You should see something like
![BentoML OpenAPI example](/doc/images/bentoml_openapi_endpoint_example.png)
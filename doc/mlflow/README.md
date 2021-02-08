# [MLFlow](https://mlflow.org)

This example shows how to install an mlflow server on Google Cloud Platform and how to
run mlflow either from your local machine or within a docker container (for example to
run mlflow in a component of a Kubeflow pipeline).

## MLFlow usage in this repository 
In this repository both the components [linear_regression_training](/components/linear_regression_training) and
[random_forest_regressor_training](/components/random_forest_regressor_training) use MLFlow to track parameters, metrics
and to log the model to the Model Registry.

The [model_loader](/components/model_loader) component shows how to load a model from the Model Registry and
[promote_model](/components/promote_model) uses MLFlow to promote a model to 'Production' by tagging it
in the Model Registry. 

## Run MLFlow locally
```
cd components/linear_regression_training
```

[Start the mlflow ui](https://mlflow.org/docs/latest/tutorials-and-examples/tutorial.html#comparing-the-models).

An MLflow tracking server has two components for storage: a backend store and an artifact store.
In order to use model registry functionality, you must run your server using a database-backed store (specified in
the following command as a sqlite db that will be placed in your current directory).
By default --backend-store-uri is set to the local ./mlruns directory. 

```
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

To run the linear_regression_training component (only) locally please comment the line
```client = storage.Client()``` in the src/model_training.py file.

Now you can run the code in your component by the following command:
```
mlflow run . -P dataset_path=/tmp/dataset.csv -P original_dataset_path=/tmp/datasets/de.csv
```

Make sure you actually have the files specified in 'dataset_path' and original_dataset_path.
As an alternative you can run the 'run_mlflow.sh' script (wich is used by the component when executed within the
Kubeflow pipeline).

Note that, to run mlflow locally, you must run the 'mlflow run' command from the same directory you have run 'mlflow ui'
(the directory containing the 'mlrun' folder).

## Run MLFLow within a docker container
As MLFlow make use of conda, to run MLFlow in a docker container you have to install conda through your Dockerfile
(as shown in the [Dockerfile](/components/linear_regression_training/Dockerfile) of the linear_regression_training
component).

You also need to install git in your Docker container. Refer to this
[Issue](https://github.com/esalvucci/mlops-architecture-example/issues/1) for further details about that.

The [run_mlflow.sh](/components/linear_regression_training/run_mlflow.sh) script is used now to run mlflow.
Such the Kubeflow pipleine must pass multiple parameters to the component wich uses mlflow, using a bash script to run
it is a workaround to manage multiple parameters be passed to the docker ENDPOINT.

## Run MLFlow project on Google Cloud Platform 
* On the Cloud side
    * Create a VM instance on Google Compute Engine and install mlflow and google cloud storage
        - Create a VM Instance
        
        ![Create Instance](/doc/images/ce_vm_create_instance.png)
        
        * Machine configuration:
            For a demo purpose it could be enough an "E2-micro" machine
        * Boot disk (I used Debian but you can choose other distros)
            * Operating System: Debian
            * Version: 10
        * Identity and API access
            * Access scopes: Allow full access to all Cloud APIs
        * Firewall
            * Allow both HTTP and HTTPS traffic
        * Network tags: http-server, https-server, <your instance name>
    
    * SSH your VM from the console and install mlflow and google cloud storage
    
    ![Open SSH](/doc/images/ce_vm_instance_ssh.png)
    
    - Install pip3, mlflow and Google Cloud Storage
        ```
        sudo apt update
        sudo apt upgrade
        # Install pip
        sudo apt install -y python3-pip
        python3 -m pip install -U six
        
        export PATH="$HOME/.local/bin:$PATH"
        # Install mlflow and google cloud storage
        pip3 install mlflow google-cloud-storage
        ```
        And check the mlflow version
        ```
        mlflow --version
        ```
    - An alternative way is to run on cli the following command
        ```
        gcloud compute instances create mlflow-server \
        --machine-type n1-standard-1 \
        --zone us-central1-a \
        --tags mlflow-server \
        --metadata startup-script='#! /bin/bash
        sudo apt update
        sudo apt-get -y install tmux
        echo Installing python3-pip
        sudo apt install -y python3-pip
        export PATH="$HOME/.local/bin:$PATH"
        echo Installing mlflow and google_cloud_storage
        pip3 install mlflow google-cloud-storage'
        ```

    * Create a firewall rule
        Click on "Setup Firewall Rules" (at the bottom of the instances table) and select "Create Firewall Rule"
        * Give it a name
        * Priority: 999
        * Direction of traffic: Ingress
        * Action on match: Allow
        * Target: Specified target tags
        * Target tags: <your instance name>
        * Source IP ranges: 0.0.0.0/0
        * Protocols and ports: select TCP with port 5000
   
        - An alternative way is to run on cli
            ```
            gcloud compute firewall-rules create mlflow-server \
            --direction=INGRESS --priority=999 --network=default \
            --action=ALLOW --rules=tcp:5000 --source-ranges=0.0.0.0/0 \
            --target-tags=mlflow-server
            ```

    * Create Cloud Storage Bucket
        ![Open SSH](/doc/images/storage_menu.png)
    
        On the left menu go to Storage and select "Create Bucket".
        
        * Access Control: "Fine grained"
        * Encryption: "Google-managed key" 

    * Launch MLFlow Server

    * Note down your instance's internal and external IP
    ![Open SSH](/doc/images/instance_ips.png)

    * Run mlflow server
    ```
    mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root gs://<bucket name> --host <internal ip>
    ```

* On your local machine

    * Set your remote tracking uri
        ```
        export MLFLOW_TRACKING_URI='http://<yout remote external ip>:5000'
        ```

    * Add Google Application Credentials
        In order for our scripts to log to the server, we need to modify our code by providing some credentials as
        environment variables.
        
        Follow the steps [here](https://cloud.google.com/iam/docs/creating-managing-service-account-keys#creating_service_account_keys)
        to create new service account key and download it locally.
        
        Now you can set the GOOGLE_APPLICATION_CREDENTIALS env variable 
        
        ```
        export GOOGLE_APPLICATION_CREDENTIALS="<your Service Account json file path>"
        ```

    * Install Google Cloud Storage
        google-cloud-storage package is required to be installed on both the client and server in order to access Google Cloud Storage
        ```
        pip3 install google-cloud-storage
        ```

    On the top (or before the "try mlflow.start") add the following lines
    ```
    from google.cloud import storage
    client = storage.Client()
    ```

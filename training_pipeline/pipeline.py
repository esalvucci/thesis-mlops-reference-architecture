import kfp
import os
from kubernetes.client import V1EnvVar


def __data_ingestion_step(dataset_name, bucket_name, folder_name):
    return kfp.dsl.ContainerOp(
            name='data_ingestion',
            image=os.environ['DOCKER_CONTAINER_REGISTRY_BASE_URL'] + '/' + os.environ['PROJECT_NAME'] + '/' +
                  os.environ['DATA_INGESTION'] + ':' + os.environ['TAG'],
            arguments=['--file_name', dataset_name,
                       '--bucket_name', bucket_name,
                       '--folder_name', folder_name],
            file_outputs={'dataset_path': '/tmp/dataset.csv'}
    )


def __data_preparation_step(dataset_path):
    return kfp.dsl.ContainerOp(
            name='data_preparation',
            image=os.environ['DOCKER_CONTAINER_REGISTRY_BASE_URL'] + '/' + os.environ['PROJECT_NAME'] + '/' +
                  os.environ['DATA_PREPARATION'] + ':' + os.environ['TAG'],
            arguments=['--dataset_path', kfp.dsl.InputArgumentPath(dataset_path)],
            file_outputs={'output_path': '/tmp/dataset.csv'}
    )


def __model_training_step(component_name, dataset_path, original_dataset_path):
    return kfp.dsl.ContainerOp(
            name=str(component_name),
            image=os.environ['DOCKER_CONTAINER_REGISTRY_BASE_URL'] + '/' + os.environ['PROJECT_NAME'] + '/' +
            str(component_name) + ':' + os.environ['TAG'],
            arguments=['--dataset_path', kfp.dsl.InputArgumentPath(dataset_path),
                       '--original_dataset_path', kfp.dsl.InputArgumentPath(original_dataset_path)],
            file_outputs={'metrics': '/tmp/metrics.json',
                          'rmse': '/tmp/rmse'}
    )


def __promotion_step(model_name):
    return kfp.dsl.ContainerOp(
            name='promote',
            image=os.environ['DOCKER_CONTAINER_REGISTRY_BASE_URL'] + '/' + os.environ['PROJECT_NAME'] + '/' +
            'promote_model' + ':' + os.environ['TAG'],
            arguments=['--model_name', model_name]
    )


@kfp.dsl.pipeline(name='Forecasting Example')
def __pipeline(training_dataset_name: str = 'it.csv',
               bucket_name: str = 'kubeflow-demo',
               folder_name: str = 'forecast-example'):
    original_dataset_path = str(os.path.join('gs://', 'kubeflow-demo', 'forecast-example', 'it.csv'))

    # Data Ingestion step
    data_ingestion = __data_ingestion_step(training_dataset_name, bucket_name, folder_name)

    # Data Preparation step
    data_preparation = __data_preparation_step(data_ingestion.output)

    # Training of the linear regression model
    sgd_regressor_component_name = 'sgd_regressor'
    linear_regression_model_training = __model_training_step(sgd_regressor_component_name,
                                                             data_preparation.output,
                                                             original_dataset_path)
    linear_regression_model_training.container.add_env_variable(V1EnvVar(name='MLFLOW_TRACKING_URI',
                                                                         value=os.environ['MLFLOW_TRACKING_URI']))

    # Training of the linear regression model
    random_forest_regressor_component_name = 'random_forest_regressor'
    random_forest_model_training = __model_training_step(random_forest_regressor_component_name,
                                                         data_preparation.output,
                                                         original_dataset_path)
    random_forest_model_training.container.add_env_variable(V1EnvVar(name='MLFLOW_TRACKING_URI',
                                                                     value=os.environ['MLFLOW_TRACKING_URI']))

    # Promote the best between the two models
    with kfp.dsl.Condition(
            random_forest_model_training.outputs['rmse'] > linear_regression_model_training.outputs['rmse']):
        __promotion_step(random_forest_regressor_component_name).container.add_env_variable(
                                                                            V1EnvVar(name='MLFLOW_TRACKING_URI',
                                                                            value=os.environ['MLFLOW_TRACKING_URI']))
    with kfp.dsl.Condition(
            random_forest_model_training.outputs['rmse'] <= linear_regression_model_training.outputs['rmse']):
        __promotion_step(sgd_regressor_component_name).container.add_env_variable(
                                                                            V1EnvVar(name='MLFLOW_TRACKING_URI',
                                                                            value=os.environ['MLFLOW_TRACKING_URI']))

#    random_forest_model_training.execution_options.caching_strategy.max_cache_staleness = "P0D"
    linear_regression_model_training.execution_options.caching_strategy.max_cache_staleness = "P0D"



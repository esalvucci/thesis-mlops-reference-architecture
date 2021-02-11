import kfp
import kfp.compiler as compiler
import kfp.dsl as dsl
from google.cloud import storage
import os
import logging
from kubernetes.client import V1EnvVar

HOST = os.environ['KUBEFLOW_HOST']
EXPERIMENT_NAME = 'Forecast Example - Training'


def run_pipeline(data, context):
    """
    Background Cloud Function to be triggered by Cloud Storage.
    This generic function logs relevant data when a file is changed, compiles the
    training Kubeflow pipelineand runs it.

    Args:
      data (dict): The Cloud Functions event payload.
      context (google.cloud.functions.Context): Metadata of triggering event.
    Returns:
      None; the output is written to Stackdriver Logging
    """
    client = kfp.Client(host=HOST)
    compiler.Compiler().compile(pipeline, '/tmp/pipeline.tar.gz')
    exp = client.create_experiment(name=EXPERIMENT_NAME)  # this is a 'get or create' op
    result = client.run_pipeline(exp.id, data['name'] + " updated", '/tmp/pipeline.tar.gz', params={})
    __log_data_info(data, context)
    logging.info(result)


@kfp.dsl.pipeline(name='Forecasting Example')
def pipeline(bucket_name: str = 'forecast-example'):
    original_dataset_path = 'gs://forecast-example/it.csv'

    # Data Ingestion step
    data_ingestion = __data_ingestion_step(bucket_name)

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

    # Avoid caching the output of the following steps
    # https://www.kubeflow.org/docs/pipelines/caching/#managing-caching-staleness
    random_forest_model_training.execution_options.caching_strategy.max_cache_staleness = "P0D"
    linear_regression_model_training.execution_options.caching_strategy.max_cache_staleness = "P0D"


def __data_ingestion_step(bucket_name):
    return kfp.dsl.ContainerOp(
            name='data_ingestion',
            image=os.environ['DOCKER_CONTAINER_REGISTRY_BASE_URL'] + '/' + os.environ['PROJECT_NAME'] + '/' +
                  os.environ['DATA_INGESTION'] + ':' + os.environ['TAG'],
            arguments=['--bucket_name', bucket_name],
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
                       '--original_dataset_path', str(original_dataset_path)],
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


def __log_data_info(data, context):
    logging.info('Event ID: {}'.format(context.event_id))
    logging.info('Event type: {}'.format(context.event_type))
    logging.info('Data: {}'.format(data))
    logging.info('Bucket: {}'.format(data['bucket']))
    logging.info('File: {}'.format(data['name']))
    file_uri = 'gs://%s/%s' % (data['bucket'], data['name'])
    logging.info('Using file uri: %s', file_uri)
    logging.info('Metageneration: {}'.format(data['metageneration']))
    logging.info('Created: {}'.format(data['timeCreated']))
    logging.info('Updated: {}'.format(data['updated']))


if __name__ == "__main__":
    run_pipeline({}, {})

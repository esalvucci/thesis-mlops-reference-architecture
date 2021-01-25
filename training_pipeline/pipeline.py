import kfp
import os
from kubernetes.client import V1EnvVar


def __data_ingestion_step(dataset_name):

    return kfp.dsl.ContainerOp(
            name='data_ingestion',
            image=os.environ['DOCKER_CONTAINER_REGISTRY_BASE_URL'] +
                  '/' + os.environ['PROJECT_NAME'] + '/' + os.environ['DATA_INGESTION'] + ':' +
                  os.environ['TAG'],
            arguments=['--file_name', dataset_name],
            file_outputs={'dataset_path': '/tmp/dataset.csv'}
    )


def __data_preparation_step(dataset_path):
    return kfp.dsl.ContainerOp(
            name='data_preparation',
            image=os.environ['DOCKER_CONTAINER_REGISTRY_BASE_URL'] +
                  '/' + os.environ['PROJECT_NAME'] + '/' + os.environ['DATA_PREPARATION'] + ':' +
                  os.environ['TAG'],
            arguments=['--dataset_path', kfp.dsl.InputArgumentPath(dataset_path)],
            file_outputs={'training_set_path': '/tmp/training_set.csv',
                          'test_set_path': '/tmp/test_set.csv'}
    )


def __model_training_step(training_set_path, test_set_path):
    return kfp.dsl.ContainerOp(
            name='model training',
            image=os.environ['DOCKER_CONTAINER_REGISTRY_BASE_URL'] +
                  '/' + os.environ['PROJECT_NAME'] + '/' + os.environ['MODEL_TRAINING'] + ':' +
                  os.environ['TAG'],
            arguments=['--training_set_path', kfp.dsl.InputArgumentPath(training_set_path),
                       '--test_set_path', kfp.dsl.InputArgumentPath(test_set_path)]
    )


@kfp.dsl.pipeline(name='Forecasting Example')
def __pipeline(training_dataset_name='it.csv'):
    data_ingestion = __data_ingestion_step(training_dataset_name)
    data_preparation = __data_preparation_step(data_ingestion.output)
    model_training = __model_training_step(data_preparation.outputs['training_set_path'],
                                           data_preparation.outputs['test_set_path'])
    model_training.container.add_env_variable(V1EnvVar(name='MLFLOW_TRACKING_URI',
                                                       value=os.environ['MLFLOW_TRACKING_URI']))
    model_training.execution_options.caching_strategy.max_cache_staleness = "P0D"



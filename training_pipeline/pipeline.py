import kfp
import os
import yaml
from kfp.v2.components import OutputPath


def __data_ingestion_step(dataset_name, dataset_path: OutputPath(str)):

    return kfp.dsl.ContainerOp(
            name='data_ingestion',
            image=os.environ['DOCKER_CONTAINER_REGISTRY_BASE_URL'] +
                  '/' + os.environ['PROJECT_NAME'] + '/' + os.environ['DATA_INGESTION'] + ':' +
                  os.environ['TAG'],
            arguments=['--file_name', dataset_name,
                       '--file_path', dataset_path],
            file_outputs={'dataset_path': '/tmp/it.csv'}
    )


def __data_preparation_step(dataset_path):
    x_training_output_path: OutputPath(str) = '/tmp/x_train.csv'
    y_training_output_path: OutputPath(str) = '/tmp/y_train.csv'
    x_test_output_path: OutputPath(str) = '/tmp/x_test.csv'
    y_test_output_path: OutputPath(str) = '/tmp/y_test.csv'

    return kfp.dsl.ContainerOp(
            name='data_preparation',
            image=os.environ['DOCKER_CONTAINER_REGISTRY_BASE_URL'] +
                  '/' + os.environ['PROJECT_NAME'] + '/' + os.environ['DATA_PREPARATION'] + ':' +
                  os.environ['TAG'],
            arguments=['--dataset_path', kfp.dsl.InputArgumentPath(dataset_path)],
            file_outputs={'x_training_set': x_training_output_path,
                          'y_training_set': y_training_output_path,
                          'x_test_set': x_test_output_path,
                          'y_test_set': y_test_output_path}
    )


def __model_training_step(dataset_path):
    return kfp.dsl.ContainerOp(
            name='model training',
            image=os.environ['DOCKER_CONTAINER_REGISTRY_BASE_URL'] +
                  '/' + os.environ['PROJECT_NAME'] + '/' + os.environ['MODEL_TRAINING'] + ':' +
                  os.environ['TAG'],
            arguments=['--dataset_path', kfp.dsl.InputArgumentPath(dataset_path)],
            file_outputs={'trained_model': '/tmp/trained_model.pkl'}
)


def __data_transformation_step(dataset_path, output_path: OutputPath(str)):
    return kfp.dsl.ContainerOp(
            name='data_transformation',
            image=os.environ['DOCKER_CONTAINER_REGISTRY_BASE_URL'] +
                  '/' + os.environ['PROJECT_NAME'] + '/' + os.environ['DATA_TRANSFORMATION'] + ':' +
                  os.environ['TAG'],
            arguments=['--dataset_path', kfp.dsl.InputArgumentPath(dataset_path),
                       '--output_path', output_path],
            file_outputs={'output': output_path}
    )


def __model_evaluation_step(dataset_name, model_path):
    return kfp.dsl.ContainerOp(
            name='model evaluation',
            image=os.environ['DOCKER_CONTAINER_REGISTRY_BASE_URL'] +
                  '/' + os.environ['PROJECT_NAME'] + '/' + os.environ['MODEL_EVALUATION'] + ':' +
                  os.environ['TAG'],
            arguments=['--dataset_name', dataset_name,
                       '--model_path', kfp.dsl.InputArgumentPath(model_path)],
            file_outputs={'accuracy': '/tmp/mean_squared_log_error.txt'}
)


@kfp.dsl.pipeline(name='Forecasting Example')
def __pipeline(training_dataset_name='it.csv', evaluation_dataset_name='de.csv'):
    dataset_path = '/tmp/it.csv'
    data_ingestion = __data_ingestion_step(training_dataset_name, dataset_path)
    data_transformation = __data_transformation_step(data_ingestion.output, dataset_path)
    model_training = __model_training_step(data_transformation.output)
    model_evaluation = __model_evaluation_step(evaluation_dataset_name, model_training.outputs['trained_model'])

    model_training.execution_options.caching_strategy.max_cache_staleness = "P0D"
    model_evaluation.execution_options.caching_strategy.max_cache_staleness = "P0D"



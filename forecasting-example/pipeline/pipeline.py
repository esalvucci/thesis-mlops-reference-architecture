import kfp
import os
import yaml
from kfp.v2.components import OutputPath
from kfp import components

CONFIG_FILENAME = '../config.yaml'
xgboost_train_on_csv_op = components.load_component_from_url('https://raw.githubusercontent.com/kubeflow/'
                                                             'pipelines/567c04c51ff00a1ee525b3458425b17adbe3df61/'
                                                             'components/XGBoost/Train/component.yaml')
xgboost_predict_on_csv_op = components.load_component_from_url('https://raw.githubusercontent.com/kubeflow/'
                                                               'pipelines/567c04c51ff00a1ee525b3458425b17adbe3df61/'
                                                               'components/XGBoost/Predict/component.yaml')
drop_header_op = kfp.components.load_component_from_url('https://raw.githubusercontent.com/kubeflow/'
                                                        'pipelines/02c9638287468c849632cf9f7885b51de4c66f86/'
                                                        'components/tables/Remove_header/component.yaml')


with open(CONFIG_FILENAME) as file:
    configuration_parameters = yaml.safe_load(file)


def __data_ingestion_step(dataset_name):

    return kfp.dsl.ContainerOp(
            name='data_ingestion',
            image=os.environ['DOCKER_CONTAINER_REGISTRY_BASE_URL'] +
                  '/' + configuration_parameters['pipeline']['name'] + '-' + 'data-ingestion:latest',
            arguments=['--dataset_name', dataset_name],
            file_outputs={'dataset_path': '/tmp/dataset.csv'}
    )


def __data_preparation_step(dataset_path):
    x_training_output_path: OutputPath(str) = '/tmp/x_train.csv'
    y_training_output_path: OutputPath(str) = '/tmp/y_train.csv'
    x_test_output_path: OutputPath(str) = '/tmp/x_test.csv'
    y_test_output_path: OutputPath(str) = '/tmp/y_test.csv'

    return kfp.dsl.ContainerOp(
            name='data_preparation',
            image=os.environ['DOCKER_CONTAINER_REGISTRY_BASE_URL'] +
                  '/' + configuration_parameters['pipeline']['name'] + '-' + 'data-preparation:latest',
            arguments=['--dataset_path', kfp.dsl.InputArgumentPath(dataset_path)],
            file_outputs={'x_training_set': x_training_output_path,
                          'y_training_set': y_training_output_path,
                          'x_test_set': x_test_output_path,
                          'y_test_set': y_test_output_path}
    )


def __model_training_step(x_training_set, y_training_set, x_test_set, y_test_set):
    return kfp.dsl.ContainerOp(
            name='model training',
            image=os.environ['DOCKER_CONTAINER_REGISTRY_BASE_URL'] +
                  '/' + configuration_parameters['pipeline']['name'] + '-' + 'model-training:latest',
            arguments=['--x_training_set_path', kfp.dsl.InputArgumentPath(x_training_set),
                       '--y_training_set_path', kfp.dsl.InputArgumentPath(y_training_set),
                       '--x_test_set_path', kfp.dsl.InputArgumentPath(x_test_set),
                       '--y_test_set_path', kfp.dsl.InputArgumentPath(y_test_set)],
            file_outputs={'trained_model': '/tmp/trained_model.pkl'}
)


def __model_evaluation_step(dataset_name, model_path):
    return kfp.dsl.ContainerOp(
            name='model evaluation',
            image=os.environ['DOCKER_CONTAINER_REGISTRY_BASE_URL'] +
                  '/' + configuration_parameters['pipeline']['name'] + '-' + 'model-evaluation:latest',
            arguments=['--dataset_name', dataset_name,
                       '--model_path', kfp.dsl.InputArgumentPath(model_path)],
            file_outputs={'accuracy': '/tmp/mean_squared_log_error.txt'}
)


def __model_prediction(input_path, output_path, model_path):

    return kfp.dsl.ContainerOp(
            name='model prediction',
            image=os.environ['DOCKER_CONTAINER_REGISTRY_BASE_URL'] +
                  '/' + configuration_parameters['pipeline']['name'] + '-' + 'model-prediction:latest',
            arguments=['--input_path', kfp.dsl.InputArgumentPath(input_path),
                       '--output_path', kfp.dsl.InputArgumentPath(output_path),
                       '--model_path', kfp.dsl.InputArgumentPath(model_path)],
            file_outputs={'output_path': output_path})


@kfp.dsl.pipeline(name='Forecasting Example')
def __pipeline(training_dataset_name='it.csv', evaluation_dataset_name='de.csv'):
    data_ingestion = __data_ingestion_step(training_dataset_name)
    drop_header = drop_header_op(data_ingestion.outputs['dataset_path'])
    model_training = xgboost_train_on_csv_op(training_data=drop_header.output,
                                             objective='reg:squarederror', num_iterations=200, label_column=1)
    model_evaluation = __model_evaluation_step(evaluation_dataset_name, model_training.outputs['model'])

    data_ingestion.execution_options.caching_strategy.max_cache_staleness = "P0D"
    model_training.execution_options.caching_strategy.max_cache_staleness = "P0D"
    model_evaluation.execution_options.caching_strategy.max_cache_staleness = "P0D"


if __name__ == '__main__':
    kfp.compiler.Compiler().compile(__pipeline, __file__ + '.yaml')

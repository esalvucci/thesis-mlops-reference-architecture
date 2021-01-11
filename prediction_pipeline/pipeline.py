import kfp
import os
import yaml
from kfp import components
from kfp.v2.components import OutputPath

xgboost_predict_on_csv_op = components.load_component_from_url('https://raw.githubusercontent.com/kubeflow/'
                                                               'pipelines/567c04c51ff00a1ee525b3458425b17adbe3df61/'
                                                               'components/XGBoost/Predict/component.yaml')
drop_header_op = components.load_component_from_url('https://raw.githubusercontent.com/kubeflow/'
                                                    'pipelines/02c9638287468c849632cf9f7885b51de4c66f86/'
                                                    'components/tables/Remove_header/component.yaml')

CONFIG_FILENAME = '../config.yaml'

with open(CONFIG_FILENAME) as file:
    configuration_parameters = yaml.safe_load(file)


def __data_ingestion_step(file_name: str, output_path: OutputPath(str)):
    return kfp.dsl.ContainerOp(
            name='data_ingestion',
            image=os.environ['DOCKER_CONTAINER_REGISTRY_BASE_URL'] +
                  '/' + configuration_parameters['pipeline']['name'] + '/' + 'data-ingestion:' +
                  os.environ['TAG'],
            arguments=['--file_name', file_name,
                       '--file_path', output_path],
            file_outputs={'output': output_path}
    )


def __data_transformation_step(dataset_path, output_path: OutputPath(str)):
    return kfp.dsl.ContainerOp(
            name='data_transformation',
            image=os.environ['DOCKER_CONTAINER_REGISTRY_BASE_URL'] +
                  '/' + configuration_parameters['pipeline']['name'] + '/' + 'data-transformation:' +
                  os.environ['TAG'],
            arguments=['--dataset_path', kfp.dsl.InputArgumentPath(dataset_path),
                       '--output_path', output_path],
            file_outputs={'output': output_path}
    )


def __prediction_step(dataset_path, output_path: OutputPath(str), model_path):
    return kfp.dsl.ContainerOp(
            name='prediction',
            image=os.environ['DOCKER_CONTAINER_REGISTRY_BASE_URL'] +
                  '/' + configuration_parameters['pipeline']['name'] + '/' + 'model-prediction:' +
                  os.environ['TAG'],
            arguments=['--dataset_path', kfp.dsl.InputArgumentPath(dataset_path),
                       '--output_path', output_path,
                       '--model_path', kfp.dsl.InputArgumentPath(model_path)],
            file_outputs={'output': output_path}
    )


@kfp.dsl.pipeline(name='Forecasting Example')
def __pipeline(prediction_dataset_name='de.csv', model_name='trained_model.pkl'):
    prediction_dataset_path = '/tmp/de.csv'
    prediction_output_path = '/tmp/de_predictions.txt'
    model_path = '/tmp/trained_model.pkl'
    data_ingestion = __data_ingestion_step(prediction_dataset_name, prediction_dataset_path)
    data_transformation = __data_transformation_step(data_ingestion.output, prediction_dataset_path)
    drop_header = drop_header_op(data_transformation.output)
    model_file = __data_ingestion_step(model_name, model_path)

    predictions = __prediction_step(drop_header.output, prediction_output_path, model_file.output)

    data_ingestion.execution_options.caching_strategy.max_cache_staleness = "P0D"
    model_file.execution_options.caching_strategy.max_cache_staleness = "P0D"
    predictions.execution_options.caching_strategy.max_cache_staleness = "P0D"


if __name__ == '__main__':
    kfp.compiler.Compiler().compile(__pipeline, __file__ + '.yaml')


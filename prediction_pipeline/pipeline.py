import kfp
import os
from kfp import components
from kubernetes.client import V1EnvVar

xgboost_predict_on_csv_op = components.load_component_from_url('https://raw.githubusercontent.com/kubeflow/'
                                                               'pipelines/master/'
                                                               'components/XGBoost/Predict/component.yaml')
drop_header_op = components.load_component_from_url('https://raw.githubusercontent.com/kubeflow/'
                                                    'pipelines/master/components/tables/Remove_header/component.yaml')


def __data_ingestion_step(file_name: str):
    return kfp.dsl.ContainerOp(
            name='data ingestion',
            image=os.environ['DOCKER_CONTAINER_REGISTRY_BASE_URL'] +
                  '/' + os.environ['PROJECT_NAME'] + '/' + os.environ['DATA_INGESTION'] +':' +
                  os.environ['TAG'],
            arguments=['--file_name', file_name],
            file_outputs={'output': '/tmp/dataset.csv'}
    )


def __load_model_step(model_name, model_stage):
    return kfp.dsl.ContainerOp(
            name='load-model',
            image=os.environ['DOCKER_CONTAINER_REGISTRY_BASE_URL'] +
                  '/' + os.environ['PROJECT_NAME'] + '/' + os.environ['MODEL_LOADER'] + ':' +
                  os.environ['TAG'],
            arguments=['--model_name', model_name,
                       '--model_stage', model_stage],
            file_outputs={'model_path': '/tmp/model/model.pkl',
                          'conda_configuration_file': '/tmp/model/conda.yaml',
                          'model_metadata': '/tmp/model/MLmodel'}
    )


def __bentoml_service(model_path, conda_configuration_file, model_metadata):
    return kfp.dsl.ContainerOp(
            name='inference-service',
            image=os.environ['DOCKER_CONTAINER_REGISTRY_BASE_URL'] +
                  '/' + os.environ['PROJECT_NAME'] + '/' + os.environ['INFERENCE_SERVICE'] + ':' +
                  os.environ['TAG'],
            arguments=['--model_path', model_path,
                       '--conda_configuration_file', conda_configuration_file,
                       '--model_metadata', model_metadata]
    )


@kfp.dsl.pipeline(name='Forecasting Example')
def __pipeline(prediction_dataset_name='de.csv', model_stage: str = 'Production'):
    model_name = 'random_forest_regressor'
    data_ingestion = __data_ingestion_step(prediction_dataset_name)
    drop_header = drop_header_op(data_ingestion.output)
    load_model = __load_model_step(model_name, model_stage)
    load_model.container.add_env_variable(V1EnvVar(name='MLFLOW_TRACKING_URI',
                                                   value=os.environ['MLFLOW_TRACKING_URI']))

    data_ingestion.execution_options.caching_strategy.max_cache_staleness = "P0D"
    load_model.execution_options.caching_strategy.max_cache_staleness = "P0D"


if __name__ == '__main__':
    kfp.compiler.Compiler().compile(__pipeline, __file__ + '.yaml')


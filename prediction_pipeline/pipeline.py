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


def __load_model_step(stage: str):
    return kfp.dsl.ContainerOp(
            name='load-model',
            image=os.environ['DOCKER_CONTAINER_REGISTRY_BASE_URL'] +
                  '/' + os.environ['PROJECT_NAME'] + '/' + os.environ['MODEL_LOADER'] + ':' +
                  os.environ['TAG'],
            arguments=['--model_name', 'xgb_regressor',
                       '--stage', stage],
            file_outputs={'output': '/tmp/model/model.xgb'}
    )


@kfp.dsl.pipeline(name='Forecasting Example')
def __pipeline(prediction_dataset_name='de.csv', model_stage: str = 'Production'):
    data_ingestion = __data_ingestion_step(prediction_dataset_name)
    drop_header = drop_header_op(data_ingestion.output)
    load_model = __load_model_step(model_stage)
    load_model.container.add_env_variable(V1EnvVar(name='MLFLOW_TRACKING_URI',
                                                       value=os.environ['MLFLOW_TRACKING_URI']))

    xgboost_predict_on_csv_op(data=drop_header.output, model=load_model.output, label_column=0)

    data_ingestion.execution_options.caching_strategy.max_cache_staleness = "P0D"
    load_model.execution_options.caching_strategy.max_cache_staleness = "P0D"


if __name__ == '__main__':
    kfp.compiler.Compiler().compile(__pipeline, __file__ + '.yaml')


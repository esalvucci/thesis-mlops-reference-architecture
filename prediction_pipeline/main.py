import kfp
import os
from kfp import components
from kubernetes.client import V1EnvVar

drop_header_op = components.load_component_from_url('https://raw.githubusercontent.com/kubeflow/'
                                                    'pipelines/master/components/tables/Remove_header/component.yaml')


@kfp.dsl.pipeline(name='Forecasting Example')
def __pipeline(bucket_name: str = 'forecast-example-prediction', model_stage: str = 'Production'):
    model_name = 'sgd_regressor'
    data_ingestion = __data_ingestion_step(bucket_name)
    data_preparation = __data_preparation_step(data_ingestion.output)
    drop_header = drop_header_op(data_preparation.output)
    load_model = __load_model_step(model_name, model_stage)
    load_model.container.add_env_variable(V1EnvVar(name='MLFLOW_TRACKING_URI',
                                                   value=os.environ['MLFLOW_TRACKING_URI']))
    inference_service = __bentoml_service(load_model.outputs['model_path'],
                                          load_model.outputs['conda_configuration_file'],
                                          load_model.outputs['model_metadata'])
    batch_prediction = __scikit_learn_batch_prediction(drop_header.output, inference_service.output)

    # Avoid caching the output of the following steps
    # https://www.kubeflow.org/docs/pipelines/caching/#managing-caching-staleness
    data_ingestion.execution_options.caching_strategy.max_cache_staleness = "P0D"
    load_model.execution_options.caching_strategy.max_cache_staleness = "P0D"
    inference_service.execution_options.caching_strategy.max_cache_staleness = "P0D"
    batch_prediction.execution_options.caching_strategy.max_cache_staleness = "P0D"


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
            name='scikit_learn_inference_service',
            image=os.environ['DOCKER_CONTAINER_REGISTRY_BASE_URL'] +
                  '/' + os.environ['PROJECT_NAME'] + '/' + os.environ['INFERENCE_SERVICE'] + ':' +
                  os.environ['TAG'],
            arguments=['--model_path', kfp.dsl.InputArgumentPath(model_path),
                       '--conda_configuration_file', kfp.dsl.InputArgumentPath(conda_configuration_file),
                       '--model_metadata', kfp.dsl.InputArgumentPath(model_metadata)],
            file_outputs={'bento_service_zip': '/tmp/bentoservice.zip'}
    )


def __scikit_learn_batch_prediction(dataset_path, bento_service_zip='/tmp/bentoservice.zip'):
    return kfp.dsl.ContainerOp(
            name='scikit_learn_batch_prediction',
            image=os.environ['DOCKER_CONTAINER_REGISTRY_BASE_URL'] +
                  '/' + os.environ['PROJECT_NAME'] + '/' + os.environ['BATCH_PREDICTION'] + ':' +
                  os.environ['TAG'],
            arguments=['--dataset_path', kfp.dsl.InputArgumentPath(dataset_path),
                       '--bento_service', kfp.dsl.InputArgumentPath(bento_service_zip)],
            file_outputs={'prediction': '/tmp/prediction.csv'}
    )


if __name__ == '__main__':
    kfp.compiler.Compiler().compile(__pipeline, __file__ + '.yaml')


import kfp
import os
import yaml

CONFIG_FILENAME = '../config.yaml'

with open(CONFIG_FILENAME) as file:
    configuration_parameters = yaml.safe_load(file)


def data_preparation_step():

    return kfp.dsl.ContainerOp(
            name='data_preparation',
            image=os.environ['DOCKER_CONTAINER_REGISTRY_BASE_URL'] +
                  '/' + configuration_parameters['pipeline']['name'] + '-' + 'data-preparation:latest',
            arguments=[],
            file_outputs={'training_set': '/tmp/training_set.png',
                          'test_set': '/tmp/test_set.png'}
    )


def model_training_step(input_data):
    return kfp.dsl.ContainerOp(
            name='model training',
            image=os.environ['DOCKER_CONTAINER_REGISTRY_BASE_URL'] +
                  '/' + configuration_parameters['pipeline']['name'] + '-' + 'model-training:latest',
            arguments=['--input_data', kfp.dsl.InputArgumentPath(input_data),
                       '--output_path', '/tmp/output.txt'],
            file_outputs={'output': '/tmp/output.txt'}
    )


@kfp.dsl.pipeline(name='Forecasting Example')
def pipeline():
    data_preparation_step()
    # model_training_step(data_preparation.outputs['data_preparation_artifact'])


if __name__ == '__main__':
    kfp.compiler.Compiler().compile(pipeline, __file__ + '.yaml')

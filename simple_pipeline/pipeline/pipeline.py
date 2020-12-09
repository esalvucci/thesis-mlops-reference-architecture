import kfp
import os
from kfp.v2.components import OutputPath


def data_preparation_step(input_dataset_url, data_preparation_output_path: OutputPath(str) = '/tmp/output.txt'):

    return kfp.dsl.ContainerOp(
            name='data_preparation',
            image=os.environ['DOCKER_CONTAINER_REGISTRY_BASE_URL'] + '/data_preparation:latest',
            arguments=['--input_dataset_url', input_dataset_url,
                       '--output_path', data_preparation_output_path],
            file_outputs={'data_preparation_artifact': data_preparation_output_path}
    )


def model_training_step(input_data):
    return kfp.dsl.ContainerOp(
            name='model training',
            image=os.environ['DOCKER_CONTAINER_REGISTRY_BASE_URL'] + '/model_training:latest',
            arguments=['--input_data', kfp.dsl.InputArgumentPath(input_data),
                       '--output_path', '/tmp/output.txt'],
            file_outputs={'output': '/tmp/output.txt'}
    )


@kfp.dsl.pipeline(name='Divina Commedia Example')
def pipeline(input_dataset_url):
    data_preparation = data_preparation_step(input_dataset_url)
    model_training_step(data_preparation.outputs['data_preparation_artifact'])


if __name__ == '__main__':
    kfp.compiler.Compiler().compile(pipeline, __file__ + '.yaml')

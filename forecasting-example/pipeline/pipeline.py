import kfp
import os

def data_preparation_step():

    return kfp.dsl.ContainerOp(
            name='data_preparation',
            image=os.environ['DOCKER_CONTAINER_REGISTRY_BASE_URL'] + '/data_preparation:latest',
            arguments=[],
            file_outputs={'training_set': '/tmp/training_set.png',
                          'test_set': '/tmp/test_set.png'}
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
def pipeline():
    data_preparation_step()
    # model_training_step(data_preparation.outputs['data_preparation_artifact'])


if __name__ == '__main__':
    kfp.compiler.Compiler().compile(pipeline, __file__ + '.yaml')

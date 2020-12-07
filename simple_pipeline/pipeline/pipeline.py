import kfp.dsl as dsl

@dsl.pipeline(name='Divina Commedia Example')
def pipeline(input_dataset_url):
    data_preparation = data_preparation_step(input_dataset_url)

def data_preparation_step(input_dataset_url):
    return dsl.ContainerOp(
            name='data_preparation',
            image='docker.io/enricosalvucci/data_preparation:latest',
            arguments=['--input_dataset_url', input_dataset_url,
                       '--output_path', '/tmp/output.txt'],
            file_outputs={'output': '/tmp/output.txt'}
    )
    
if __name__ == '__main__':
    kfp.compiler.Compiler().compile(pipeline, __file__ + '.yaml')  

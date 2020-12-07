import kfp.dsl as dsl

@dsl.pipeline()
def pipeline():
    data_preparation = data_preparation_step()

def data_preparation_step():
    return dsl.ContainerOp(
            name='data_preparation',
            image='docker.io/enricosalvucci/pipeline-example:latest',
            arguments=['--output_path', '/tmp/output.txt'],
            file_outputs={'output': '/tmp/output.txt'}
    )
    
if __name__ == '__main__':
    kfp.compiler.Compiler().compile(pipeline, __file__ + '.yaml')  

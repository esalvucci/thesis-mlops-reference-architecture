import fire
import mlflow
from mlflow.exceptions import RestException
from utility.singleton_logger import SingletonLogger

logger = SingletonLogger.get_logger()


def load(model_name, stage='Production'):
    """
    Loads a model from the MLFlow Model Registry at the URI specified in the MLFLOW_TRACKING_URI env variable and
    save it in a local directory.
    :param model_name - The name of the model to be retreived
    :param stage - The stage of the model to be retreived, either 'Staging' or 'Production'
    :param experiment_name - The experiment name
    """
    output_path = '/tmp/model'
    try:
        model = mlflow.sklearn.load_model(model_uri=f"models:/{model_name}/{stage}")
        mlflow.sklearn.save_model(model, output_path)
        logger.info("Model loaded and saved in " + output_path)
    except RestException:
        logger.error("The model with name " + model_name + " in stage " + stage + " has not been found")


if __name__ == "__main__":
    fire.Fire(load)

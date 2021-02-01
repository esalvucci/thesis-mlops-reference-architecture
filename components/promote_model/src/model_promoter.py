import fire
import mlflow
from mlflow.tracking import MlflowClient
from utility.singleton_logger import SingletonLogger

logger = SingletonLogger.get_logger()


def __get_last_version_number(model_name):
    client = MlflowClient()
    versions = list()
    filter_string = "name='" + model_name + "'"
    for model_version in client.search_model_versions(filter_string):
        versions.append(int(model_version.version))
    logger.info(model_name + " - Last version " + str(max(versions)))
    return max(versions)


def promote(model_name, stage='Production'):
    """
    Loads a model from the MLFlow Model Registry at the URI specified in the MLFLOW_TRACKING_URI env variable.
    :param model_name - The name of the model to be promoted
    :param stage - The stage of the model to be promoted, either 'Staging' or 'Production'
    """
    client = MlflowClient()
    version = __get_last_version_number(model_name)
    client.transition_model_version_stage(model_name, version=version, stage=stage)


if __name__ == "__main__":
    fire.Fire(promote)

import fire
import mlflow
from singleton_logger import SingletonLogger

logger = SingletonLogger.get_logger()


def __get_mlflow_experiment(name):
    if not mlflow.get_experiment_by_name(name):
        raise BaseException("The experiment does not exists")
    return mlflow.get_experiment_by_name(name)


def load(model_name, stage):
    experiment = __get_mlflow_experiment("Default")
    with mlflow.start_run(experiment_id=experiment.experiment_id):
        output_path = '/tmp/model'
        model = mlflow.xgboost.load_model(model_uri=f"models:/{model_name}/{stage}")
        mlflow.xgboost.save_model(model, output_path)
        logger.info("Model loaded and saved in " + output_path)


if __name__ == "__main__":
    fire.Fire(load)

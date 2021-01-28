import mlflow
import pandas as pd
from singleton_logger import SingletonLogger

logger = SingletonLogger.get_logger()


def __get_mlflow_experiment(name):
    if not mlflow.get_experiment_by_name(name):
        raise BaseException("The experiment does not exists")
    return mlflow.get_experiment_by_name(name)


if __name__ == "__main__":
    df = pd.read_csv('/tmp/test.csv')
    model_name = "random_forest_regressor"
    stage = "Production"
    experiment = __get_mlflow_experiment("Default")
    with mlflow.start_run(experiment_id=experiment.experiment_id):
        model = mlflow.sklearn.load_model(model_uri=f"models:/{model_name}/{stage}")
        logger.info(model.attributes())
        predictions = model.predict(df)
        print(predictions)

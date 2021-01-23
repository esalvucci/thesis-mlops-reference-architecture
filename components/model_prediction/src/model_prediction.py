import fire
import pandas as pd
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient
from singleton_logger import SingletonLogger
from xgboost import DMatrix

logger = SingletonLogger.get_logger()
model_name = "xgb_regressor"


def __get_mlflow_experiment(name):
    if not mlflow.get_experiment_by_name(name):
        raise BaseException("The experiment does not exists")
    return mlflow.get_experiment_by_name(name)


def __get_last_version_model():
    def get_last_version_number():
        client = MlflowClient()
        versions = list()
        filter_string = "name='" + model_name + "'"
        for model_version in client.search_model_versions(filter_string):
            versions.append(model_version.version)
        logger.info("Last version " + max(versions))
        return max(versions)
    version_number = get_last_version_number()
    return mlflow.xgboost.load_model(model_uri=f"models:/{model_name}/{version_number}")


def __split_data_into_x_y(data):
    target_col = 0
    x = data.drop(columns=data.columns[target_col], axis=1)
    y = data.loc[:, data.columns[target_col]]
    return x, y


def __save_predictions_at(output_path, predictions):
    np.savetxt(output_path, predictions, delimiter=',')


def predict(dataset_path, output_path):
    df = pd.read_csv(dataset_path)
    df = df.drop(columns="end").set_index("start")
    df = df.dropna()

    df.index = pd.to_datetime(df.index, errors='coerce')
    df.index.name = "time"

    x_dataset = df.drop(columns='load')
    y_dataset = df.loc[:, 'load']

    predictions_dataset = DMatrix(x_dataset, y_dataset)
    experiment = __get_mlflow_experiment("Default")
    with mlflow.start_run(experiment_id=experiment.experiment_id):
        model = __get_last_version_model()
        predictions = model.predict(predictions_dataset)
        __save_predictions_at(output_path, predictions)
        logger.info("Prediction output saved at " + output_path)


if __name__ == "__main__":
    fire.Fire(predict)

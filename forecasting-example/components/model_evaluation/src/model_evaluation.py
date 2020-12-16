from google.cloud import storage
import pandas as pd
import os
import fire
import sklearn.metrics as metrics
from xgboost import XGBRegressor, Booster
from singleton_logger import SingletonLogger

logger = SingletonLogger.get_logger()


def evaluate_model(dataset_name, model_path):
    bucket_name = 'kubeflow-demo'
    folder_path = 'forecast-example'
    dataset_path = os.path.join('/tmp/', dataset_name)
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    bucket.blob(os.path.join(folder_path, dataset_name)).download_to_filename(dataset_path)

    df = pd.read_csv(dataset_path)
    df = df.drop(columns="end").set_index("start")
    df = df.dropna()

    df.index = pd.to_datetime(df.index, errors='coerce')
    df.index.name = "time"

    x_dataset = df.drop(columns='load')
    y_dataset = df.loc[:, 'load']

    model = XGBRegressor()
    booster = Booster()
    booster.load_model(model_path)
    model._Booster = booster

    predictions = model.predict(x_dataset)

    mean_squared_log_error = metrics.mean_squared_log_error(y_dataset, predictions)

    with open('/tmp/mean_squared_log_error.txt', 'w') as output_text:
        mean_squared_log_error = str(mean_squared_log_error)
        output_text.write(mean_squared_log_error)

    logger.info("Mean squared error " + mean_squared_log_error)


if __name__ == "__main__":
    fire.Fire(evaluate_model)

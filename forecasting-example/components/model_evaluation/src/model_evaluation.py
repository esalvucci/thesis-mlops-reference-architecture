from google.cloud import storage
import pandas as pd
import os
import fire
import sklearn.metrics as metrics
import pickle
from singleton_logger import SingletonLogger

logger = SingletonLogger.get_logger()


def evaluate_model(dataset_name, model_path):
    bucket_name = 'kubeflow-demo'
    folder_path = 'forecast-example'
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    bucket.blob(os.path.join(folder_path, dataset_name)).download_to_filename(dataset_name)

    df = pd.read_csv(dataset_name)
    df = df.drop(columns="end").set_index("start")
    df = df.dropna()

    df.index = pd.to_datetime(df.index, errors='coerce')
    df.index.name = "time"

    x_dataset = df.drop(columns='load')
    y_dataset = df.loc[:, 'load']

    try:
        with open(model_path, 'rb') as model_file:
            model = pickle.load(open(os.path.join(folder_path, model_path), 'rb'))
            predictions = model.predict(x_dataset)

            mean_squared_log_error = metrics.mean_squared_log_error(y_dataset, predictions)
            logger.info("mean_squared_log_error: " + str(mean_squared_log_error))

            with open('/tmp/mean_squared_log_error.txt', 'w') as output_text:
                output_text.write(str(mean_squared_log_error))
                logger.info("Metric saved")
    except FileNotFoundError:
        logger.info("The file in the specified model path does not exists")

if __name__ == "__main__":
    fire.Fire(evaluate_model)

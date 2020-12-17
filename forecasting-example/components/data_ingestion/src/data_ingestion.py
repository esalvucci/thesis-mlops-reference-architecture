import fire
import os
import pandas as pd
from google.cloud import storage
from singleton_logger import SingletonLogger

logger = SingletonLogger.get_logger()


def __get_data(dataset_name):
    bucket_name = 'kubeflow-demo'
    folder_path = 'forecast-example'
    dataset_path = '/tmp/dataset.csv'
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    bucket.blob(os.path.join(folder_path, dataset_name)).download_to_filename(dataset_path)

    df = pd.read_csv(dataset_path)
    df = df.drop(columns="end").drop(columns="start")
    df.index.name = 'index'

    df.to_csv(dataset_path)
    logger.info("Dataset saved in " + dataset_path)


if __name__ == "__main__":
    fire.Fire(__get_data)

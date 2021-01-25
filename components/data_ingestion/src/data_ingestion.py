import fire
import os
from google.cloud import storage
import pandas as pd
from singleton_logger import SingletonLogger

logger = SingletonLogger.get_logger()


def __get_data(file_name):
    bucket_name = 'kubeflow-demo'
    folder_path = 'forecast-example'
    output_path = '/tmp/dataset.csv'
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    bucket.blob(os.path.join(folder_path, file_name)).download_to_filename(output_path)

    df = pd.read_csv(output_path)
    df = df.drop(columns="end").set_index("start")
    df.index = pd.to_datetime(df.index)
    df = df.groupby(pd.Grouper(freq="h")).mean()
    df.index.name = "time"
    df.to_csv(output_path)


if __name__ == "__main__":
    fire.Fire(__get_data)

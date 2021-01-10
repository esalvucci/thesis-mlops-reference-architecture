import fire
import os
from google.cloud import storage
from singleton_logger import SingletonLogger

logger = SingletonLogger.get_logger()


def __get_data(file_name, file_path):
    bucket_name = 'kubeflow-demo'
    folder_path = 'forecast-example'
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    bucket.blob(os.path.join(folder_path, file_name)).download_to_filename(file_path)


if __name__ == "__main__":
    fire.Fire(__get_data)

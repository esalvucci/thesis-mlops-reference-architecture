import fire
import os
from google.cloud import storage
from google.api_core.exceptions import NotFound
from utility.singleton_logger import SingletonLogger

logger = SingletonLogger.get_logger()


def get_data(file_name, bucket_name, folder_name):
    """
    Retrieves the dataset from a bucket on Google Cloud Storage

    gs://<bucket_name>/<folder_name>/<file_name>

    e.g. gs://kubeflow-demo/forecast-example/it.csv

    :param file_name - The name of the file to be retreived
    :param bucket_name - The name of the bucket
    :param folder_name - The folder in which the file is located within the bucket
    """
    output_path = '/tmp/dataset.csv'
    client = storage.Client()
    try:
        bucket = client.get_bucket(bucket_name)
        bucket.blob(os.path.join(folder_name, file_name)).download_to_filename(output_path)
        logger.info("Data saved in " + output_path)
    except NotFound:
        logger.error("File or Bucket have not been found")


if __name__ == "__main__":
    fire.Fire(get_data)

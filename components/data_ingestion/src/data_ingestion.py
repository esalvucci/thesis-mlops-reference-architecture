import fire
import glob
import os
import pandas as pd
from google.cloud import storage
from google.api_core.exceptions import NotFound
from utility.singleton_logger import SingletonLogger
import numpy as np

logger = SingletonLogger.get_logger()
columns = pd.Index(['start', 'end', 'load'], dtype=bool)


def get_all_files_in_bucket(bucket_name):
    storage_client = storage.Client()
    # Note: Client.list_blobs requires at least package version 1.17.0.
    return storage_client.list_blobs(bucket_name)


def dataset_columns_match_with(df_columns):
    return np.all(df_columns, columns)


def get_all_feasible_files_in(directory):
    files = glob.glob(directory + "*.csv")
    feasible_files = []
    for file in files:
        df = pd.read_csv(file, nrows=1)
        if dataset_columns_match_with(df.columns):
            feasible_files.append(file)
    return feasible_files


def get_df_from_files_in(files_list):
    df_list = []
    for filename in files_list:
        df_list.append(pd.read_csv(filename))
    return pd.concat(df_list)


def remove_imported_files_from(directory):
    for file in get_all_feasible_files_in(directory):
        os.remove(file)


def remove_files_in(folder):
    files = glob.glob(folder + "*.csv")
    for f in files:
        os.remove(f)


def get_data(bucket_name):
    """
    Retrieves the dataset from a bucket on Google Cloud Storage

    gs://<bucket_name/

    :param bucket_name - The name of the bucket
    """
    local_folder_path = '/tmp/'
    remove_files_in(local_folder_path)
    try:
        blobs = get_all_files_in_bucket(bucket_name)
        for b in blobs:
            name = b.name
            b.download_to_filename(local_folder_path + name)
        df = get_df_from_files_in(get_all_feasible_files_in(local_folder_path))
        remove_imported_files_from(local_folder_path)
        df.sort_values(by=['start'], inplace=True, ascending=True)

        output_path = local_folder_path + 'dataset.csv'
        df.to_csv(output_path, index=False)
        logger.info("Data saved in " + output_path)
    except NotFound:
        logger.error("File or Bucket have not been found")


if __name__ == "__main__":
    fire.Fire(get_data)

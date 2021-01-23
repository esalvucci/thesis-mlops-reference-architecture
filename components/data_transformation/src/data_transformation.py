import pandas as pd
from singleton_logger import SingletonLogger
import fire

logger = SingletonLogger.get_logger()


def __transform_data(dataset_path, output_path='/tmp/transformed_data.csv'):
    df = pd.read_csv(dataset_path)
    df = df.drop(columns="end").drop(columns="start")
    df.dropna(inplace=True)
    df.index.name = 'index'
    df.to_csv(output_path)
    logger.info("Dataset saved in " + output_path)


if __name__ == "__main__":
    fire.Fire(__transform_data)

import pandas as pd
from electricity_consumption_dataset import ElectricityConsumptionDataset
from utility.singleton_logger import SingletonLogger
import fire

logger = SingletonLogger.get_logger()


def prepare_data(dataset_path):
    """
    Prepares the dataset to be used in the model training phase
    :param dataset_path - The path for the incoming dataset
    :return:
    """
    output_path = '/tmp/dataset.csv'
    df = pd.read_csv(dataset_path)
    df = df.drop(columns="end").set_index("start")
    df.index = pd.to_datetime(df.index)
    df = df.groupby(pd.Grouper(freq="h")).mean()
    df.index.name = "time"
    electricity_consumption_dataset = ElectricityConsumptionDataset(df)
    df = electricity_consumption_dataset.get_transformed_dataset()
    logger.info(df)
    df.to_csv(output_path)
    logger.info("Dataset saved in " + output_path)


if __name__ == "__main__":
    fire.Fire(prepare_data)

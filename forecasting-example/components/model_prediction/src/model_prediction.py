import fire
import pandas as pd
import pickle
import numpy as np
from xgboost import XGBRegressor, Booster

from singleton_logger import SingletonLogger

logger = SingletonLogger.get_logger()


def __get_model_from(model_path):
    try:
        model = XGBRegressor()
        booster = Booster()
        booster.load_model(model_path)
        model._Booster = booster
        return model
    except FileNotFoundError:
        logger.info("The file in the specified model path does not exists")


def __save_predictions_at(output_path, predictions):
    with open(output_path, 'wb') as output_text:
        np.savetxt(output_text, predictions, delimiter=',')


def predict(dataset_input_path, output_path, model_path):
    model = __get_model_from(model_path)

    try:
        x = pd.read_csv(dataset_input_path)
        x = x.drop(columns="end").set_index("start")
        x = x.dropna()

        x.index = pd.to_datetime(x.index, errors='coerce')
        x.index.name = "time"

        predictions = model.predict(x)
        __save_predictions_at(output_path, predictions)
        logger.info("Prediction output saved at " + output_path)

    except FileNotFoundError:
        logger.info("The file in the specified input_path does not exists")


if __name__ == "__main__":
    fire.Fire(predict)

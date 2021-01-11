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


def __split_data_into_x_y(data):
    target_col = "index"
    x = data.drop(columns=target_col)
    y = data.loc[:, target_col]
    return x, y


def __save_predictions_at(output_path, predictions):
    with open(output_path, 'wb') as output_text:
        np.savetxt(output_text, predictions, delimiter=',')


def predict(input_path, output_path, model_path):
    model = __get_model_from(model_path)

    try:
        df = pd.read_csv(input_path)
        df = df.dropna()

        x, y = __split_data_into_x_y(df)

        predictions = model.predict(x)
        __save_predictions_at(output_path, predictions)
        logger.info("Prediction output saved at " + output_path)

    except FileNotFoundError:
        logger.info("The file in the specified input_path does not exists")


if __name__ == "__main__":
    fire.Fire(predict)

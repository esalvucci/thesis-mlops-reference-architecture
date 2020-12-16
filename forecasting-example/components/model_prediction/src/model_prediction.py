import fire
import pandas as pd
import logging
from xgboost import XGBRegressor, Booster
import numpy as np


def predict(input_path, output_path, model_path):
    model = XGBRegressor()
    booster = Booster()
    booster.load_model(model_path)
    model._Booster = booster

    x = pd.read_csv(input_path, index_col='time')
    predictions = model.predict(x)
    logging.info(predictions)

    with open(output_path, 'wb') as output_text:
        np.savetxt(output_text, predictions, delimiter=',')


if __name__ == "__main__":
    fire.Fire(predict)

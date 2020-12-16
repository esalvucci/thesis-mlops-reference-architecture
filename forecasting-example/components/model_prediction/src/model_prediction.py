import fire
import pandas as pd
import pickle
import numpy as np
from singleton_logger import SingletonLogger

logger = SingletonLogger.get_logger()


def predict(input_path, output_path, model_path):
    try:
        with open(model_path, 'rb') as model_file:
            model = pickle.load(model_file)

            x = pd.read_csv(input_path, index_col='time')
            predictions = model.predict(x)

            with open(output_path, 'wb') as output_text:
                np.savetxt(output_text, predictions, delimiter=',')
                logger.info("Prediction output saved at " + output_path)

    except FileNotFoundError:
        logger.info("The file in the specified model path does not exists")


if __name__ == "__main__":
    fire.Fire(predict)

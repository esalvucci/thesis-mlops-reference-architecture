from xgboost import XGBRegressor
import pandas as pd
import fire
from singleton_logger import SingletonLogger

logger = SingletonLogger.get_logger()


def __save(model):
    model.save_model('/tmp/trained_model.pkl')
    logger.info("Model saved")


def __split_train_test(df, split_percent=0.66):
    split = len(df) * split_percent
    train_set = df.loc[df.index < split]
    test_set = df.loc[df.index > split]
    return train_set, test_set


def __split_data_into_x_y(data):
    target_col = "index"
    x = data.drop(columns=target_col)
    y = data.loc[:, target_col]
    return x, y


def train_model(dataset_path):
    n_estimators = 1000
    learning_rate = 0.5
    max_depth = 5

    dataset = pd.read_csv(dataset_path)
    train_set, test_set = __split_train_test(dataset)
    x_training_set, y_training_set = __split_data_into_x_y(train_set)
    x_test_set, y_test_set = __split_data_into_x_y(test_set)

    model = XGBRegressor(
            n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, min_child_weight=5
    )

    model.fit(
        x_training_set, y_training_set, early_stopping_rounds=10,
        eval_set=[(x_training_set, y_training_set), (x_test_set, y_test_set)],
        verbose=False,
    )

    __save(model)


if __name__ == "__main__":
    fire.Fire(train_model)

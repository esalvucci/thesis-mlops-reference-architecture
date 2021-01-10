import pandas as pd
import numpy as np
import holidays
import matplotlib.pyplot as plt
from sklearn.preprocessing import (StandardScaler, OneHotEncoder, FunctionTransformer)
from sklearn.compose import ColumnTransformer
from singleton_logger import SingletonLogger
import fire

logger = SingletonLogger.get_logger()
STUDY_START_DATE = pd.Timestamp("2015-01-01 00:00", tz="utc")
STUDY_END_DATE = pd.Timestamp("2020-01-31 23:00", tz="utc")


def __prepare_data(dataset_path):
    df = pd.read_csv(dataset_path)
    df = df.drop(columns="end").set_index("start")
    df.index = pd.to_datetime(df.index, errors='coerce')
    df.index.name = "time"
    df = df.groupby(pd.Grouper(freq="h")).mean()
    df = df.loc[
         (df.index >= STUDY_START_DATE) & (df.index <= STUDY_END_DATE), :
         ]

    df_train, df_test = __split_train_test(
        df, pd.Timestamp("2019-02-01", tz="utc")
    )

    __save_train_test_set_artifacts(df_train, df_test)

    var = df_train.loc[df_train["load"].isna(), :].index
    df_train = __add_all_features(df_train).dropna()
    df_test = __add_all_features(df_test).dropna()

    x_train, y_train = __split_data_into_x_y(df_train)
    x_test, y_test = __split_data_into_x_y(df_test)

    feature_names, prep_pipeline = __fit_prep_pipeline(x_train)

    x_train_prep = prep_pipeline.transform(x_train)
    x_train_prep = pd.DataFrame(x_train_prep, columns=feature_names, index=df_train.index)

    x_test_prep = prep_pipeline.transform(x_test)
    x_test_prep = pd.DataFrame(x_test_prep, columns=feature_names, index=df_test.index)

    x_train_prep.to_csv('/tmp/x_train.csv')
    logger.info("X Training set saved")

    y_train.to_csv('/tmp/y_train.csv')
    logger.info("Y Training set saved")

    x_test_prep.to_csv('/tmp/x_test.csv')
    logger.info("X Test set saved")

    y_test.to_csv('/tmp/y_test.csv')
    logger.info("Y Test set saved")


if __name__ == "__main__":
    fire.Fire(__prepare_data)

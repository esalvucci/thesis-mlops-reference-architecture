import pandas as pd
import numpy as np
from utility.singleton_logger import SingletonLogger
import fire
import holidays

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

    test_set = __add_all_features(df).dropna()
    test_set.to_csv(output_path)
    logger.info("Dataset saved in " + output_path)


def __add_time_features(df):
    cet_index = df.index.tz_convert("CET")
    df["month"] = cet_index.month
    df["weekday"] = cet_index.weekday
    df["hour"] = cet_index.hour
    return df


def __add_holiday_features(df):
    df.index = pd.to_datetime(df.index)
    df_holidays = holidays.Germany()
    cet_dates = pd.Series(df.index.tz_convert("CET"), index=df.index)
    df["holiday"] = cet_dates.apply(lambda d: d in df_holidays)
    df["holiday"] = df["holiday"].astype(int)
    return df


def __add_lag_features(df, col="load"):
    for n_hours in range(24, 49):
        shifted_col = df[col].shift(n_hours, "h")
        shifted_col = shifted_col.loc[df.index.min(): df.index.max()]
        label = f"{col}_lag_{n_hours}"
        df[label] = np.nan
        df.loc[shifted_col.index, label] = shifted_col
    return df


def __add_all_features(df, target_col="load"):
    df = df.copy()
    df = __add_time_features(df)
    df = __add_holiday_features(df)
    df = __add_lag_features(df, col=target_col)
    return df


def __split_train_test(df, split_time):
    df_train = df.loc[df.index < split_time]
    df_test = df.loc[df.index >= split_time]
    return df_train, df_test


if __name__ == "__main__":
    fire.Fire(prepare_data)
